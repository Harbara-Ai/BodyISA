import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from bodyisa.bfm_adapter import BFMAdapter,ReferenceMotion,MINIMAL
from bodyisa.core import CapabilityRegistry,ValidationEnvelope
from bodyisa.phase3 import (make_manifest,policy_history_fingerprint,live_state_fingerprint,
    LiveMutationGuard,reference_entry_mismatch,verify_handoff,aggregate,canonical_hash)


ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'tests/fixtures/phase3_config.json').read_text())


class Phase3Tests(unittest.TestCase):
    def test_same_seed_rebuilds_identical_manifest(self):
        first=make_manifest(CONFIG)
        self.assertEqual(first,make_manifest(CONFIG))
        self.assertEqual(len(first),sum(CONFIG['trials'].values()))
        self.assertEqual(len({r['trial_id'] for r in first}),len(first))
        changed=copy.deepcopy(CONFIG);changed['seeds']['exploration']+=1
        self.assertNotEqual(canonical_hash(first),canonical_hash(make_manifest(changed)))

    def test_episode_guard_forbids_live_reset_and_set_state(self):
        a=BFMAdapter(device='cpu')
        guard=LiveMutationGuard(a)
        before=live_state_fingerprint(a)
        guard.locked=True
        with self.assertRaisesRegex(RuntimeError,'reset forbidden'): a.env.reset()
        with self.assertRaisesRegex(RuntimeError,'set_state forbidden'):
            a.env.set_state(np.zeros(29))
        self.assertEqual(live_state_fingerprint(a),before)
        self.assertEqual(guard.calls['blocked_reset'],1)
        self.assertEqual(guard.calls['blocked_set_state'],1)
        guard.unlock_and_restore()

    def test_handoff_requires_exact_source_terminal_and_history(self):
        a=BFMAdapter(device='cpu')
        source=a.snapshot()
        entry=a.snapshot()
        history=policy_history_fingerprint(a)
        good=verify_handoff(source,entry,history,history)
        self.assertTrue(all(good.values()))
        altered=replace(entry,base_position_world=(1,0,entry.root_height))
        self.assertFalse(verify_handoff(source,altered,history,history)['source_target_snapshot_equal_excluding_wall_time'])
        fake_history={**history,'sha256':'cleared'}
        self.assertFalse(verify_handoff(source,entry,history,fake_history)['source_target_history_equal'])

    def test_reference_mismatch_measurement_does_not_align_live_robot(self):
        a=BFMAdapter(device='cpu')
        a.reference=ReferenceMotion(MINIMAL/'example_motion/dance1_subject2_50_jpos.npz',500)
        before=live_state_fingerprint(a)
        history=policy_history_fingerprint(a)
        mismatch=reference_entry_mismatch(a,a.snapshot())
        self.assertGreaterEqual(mismatch['joint_rmse_rad'],0)
        self.assertEqual(live_state_fingerprint(a),before)
        self.assertEqual(policy_history_fingerprint(a),history)

    def test_capability_requires_matching_scope_contract_asset_and_backend(self):
        c=CapabilityRegistry();c.require_envelope=True
        c.add('TRACK_MOTION',['API_AVAILABLE','ASSET_AVAILABLE'])
        base=dict(behavior='TRACK_MOTION',target_id='dance_prefix_500',asset_sha256='asset',
            checkpoint_sha256='model',simulator_version='3.3.7',adapter_version='v1',
            contract_hash='contract',scope='SIM_ONLY',candidate_region={},empirical_support=({'trial_id':'one'},),
            runtime_entry_conditions={},initialization='reference_first',previous_behavior=None,
            policy_history='RESET_ACTOR_HISTORY',reference_alignment='FIXED_WORLD_V1',split='heldout_validation',
            counts={'contract_success':1},failure_breakdown={},terminal_errors={},completion_times_s={},
            quality_metrics={},evidence_paths=('trial.json',))
        c.record_envelope(ValidationEnvelope(**base))
        identity=dict(checkpoint_sha256='model',mujoco='3.3.7',adapter_version='v1')
        with self.assertRaisesRegex(RuntimeError,'LEGACY_TRACK_SEMANTICS_WITHHELD'):
            c.authorize('TRACK_MOTION','contract','dance_prefix_500',identity,'asset')
        self.assertEqual(c.describe_capabilities(),[])
        c.withhold('TRACK_MOTION','Negative control passes legacy completion')
        self.assertEqual(c.describe_capabilities(),[])
        with self.assertRaisesRegex(RuntimeError,'CAPABILITY_WITHHELD'):
            c.authorize('TRACK_MOTION','contract','dance_prefix_500',identity,'asset')
        c.entries['TRACK_MOTION']['exposure_status']='EXPOSABLE'
        for mismatch in [('wrong','dance_prefix_500',identity,'asset'),
                         ('contract','other',identity,'asset'),
                         ('contract','dance_prefix_500',identity,'other'),
                         ('contract','dance_prefix_500',{**identity,'adapter_version':'v2'},'asset'),
                         ('contract','dance_prefix_500',{**identity,'checkpoint_sha256':'other'},'asset')]:
            with self.assertRaisesRegex(RuntimeError,'NO_MATCHING_VALIDATION_ENVELOPE'):
                c.authorize('TRACK_MOTION',*mismatch,semantic_profile='JOINT_TRAJECTORY_V2')
        no_scope=CapabilityRegistry();no_scope.require_envelope=True
        no_scope.add('TRACK_MOTION',['API_AVAILABLE'])
        no_scope.record_envelope(ValidationEnvelope(**{**base,'scope':'HARDWARE'}))
        self.assertEqual(no_scope.describe_capabilities(),[])

    def test_sequence_failure_and_accounting_categories_do_not_become_success(self):
        manifest=make_manifest(CONFIG)
        rows=[next(t for t in manifest if t['group']==g) for g in
              ('composition_s3','stand_exploration','track_exploration','transition_s1')]
        results={rows[0]['trial_id']:dict(status='EXECUTION_FAILED',failure_stage='TRACK_2',failure_reason='TIMEOUT'),
                 rows[1]['trial_id']:dict(status='INVALID_INITIALIZATION',failure_reason='JOINT_LIMIT'),
                 rows[2]['trial_id']:dict(status='HARNESS_ERROR',failure_reason='SCRIPT_BUG'),
                 rows[3]['trial_id']:dict(status='PRECONDITION_REJECTED',failure_stage='TRACK_1',failure_reason='PRECONDITION_REJECTED')}
        agg=aggregate(CONFIG,manifest,results)
        self.assertEqual(agg['composition_s3']['contract_success'],0)
        self.assertEqual(agg['composition_s3']['timeout'],1)
        self.assertEqual(agg['composition_s3']['first_failure_stage'],{'TRACK_2':1})
        self.assertEqual(agg['stand_exploration']['invalid_initialization'],1)
        self.assertEqual(agg['track_exploration']['harness_error'],1)
        self.assertEqual(agg['transition_s1']['precondition_rejected'],1)

    def test_reference_timing_and_phase_separation(self):
        ref=ReferenceMotion(MINIMAL/'example_motion/dance1_subject2_50_jpos.npz',500)
        self.assertAlmostEqual(ref.duration,9.98)
        self.assertEqual(CONFIG['reference_preprocessing']['alignment_freeze'],
            'original frame for full episode; no per-step realignment')
        a=BFMAdapter(device='cpu')
        self.assertAlmostEqual(a.dt,.02)
        self.assertEqual(round(ref.duration/a.dt),499)
        j_before,_,_=ref.sample(ref.duration-.01)
        j_end,_,_=ref.sample(ref.duration)
        self.assertFalse(np.array_equal(j_before,j_end))


if __name__=='__main__':unittest.main()
