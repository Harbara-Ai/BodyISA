import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
import numpy as np
from bodyisa.track_contract_v2 import JointTrajectoryObserver
from bodyisa.core import (CapabilityRegistry,ExecutionRuntime,Instruction,BodyObjective,
                          issue_local_evaluation_permit,ValidationEnvelope,StateSnapshot)
from bodyisa.bfm_adapter import BFMAdapter

ROOT=Path(__file__).resolve().parents[1]
V2=json.loads((ROOT/'bodyisa/profiles/g1/track_v2.json').read_text())
PHYSICAL=json.loads((ROOT/'bodyisa/profiles/g1/physical.json').read_text())


class Reference:
    duration=9.98
    reference_id='dance_prefix_500'
    source_hash='asset'
    def __init__(self):
        self.joints=np.zeros((500,29))
        self.joints[:,0]=np.sin(np.arange(500)*.02)
        self.joints[:,1]=np.cos(np.arange(500)*.02)


def score(values,times=None,config=V2):
    observer=JointTrajectoryObserver(Reference(),config)
    times=np.arange(len(values))*.02 if times is None else times
    for q,t in zip(values,times):
        observer.observe(type('S',(),{'joint_position':q})(),t)
    return observer.finalize()


class Phase31Tests(unittest.TestCase):
    def test_perfect_and_provenance_independent(self):
        trace=Reference().joints.copy()
        self.assertEqual(score(trace)['status'],'PASS')
        self.assertEqual(score(trace),score(trace.copy()))

    def test_stand_and_sustained_deviation(self):
        self.assertEqual(score(np.zeros((500,29)))['status'],'FAIL')
        trace=Reference().joints.copy();trace[100:115]+=1.5
        result=score(trace)
        self.assertEqual(result['status'],'FAIL')
        self.assertLess(result['failure']['first_violation_time_s'],2.3)

    def test_time_coverage_duplicate_gap_shift_final(self):
        trace=Reference().joints.copy()
        for times in (np.r_[0,np.arange(1,500)*.02+.02],
                      np.r_[0,0,np.arange(2,500)*.02],
                      np.r_[np.arange(499)*.02,10.]):
            self.assertEqual(score(trace,times)['status'],'NOT_EVALUABLE')
        self.assertEqual(score(trace[:-1])['status'],'NOT_EVALUABLE')
        self.assertEqual(score(trace,np.arange(500)*.02)['status'],'PASS')

    def test_nonfinite_and_parity(self):
        trace=Reference().joints.copy();trace[200,0]=np.nan
        self.assertEqual(score(trace)['status'],'NOT_EVALUABLE')
        bad_reference=Reference();bad_reference.joints[20,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite reference'):
            JointTrajectoryObserver(bad_reference,V2)
        trace=Reference().joints.copy();trace[120:150]+=1
        offline=score(trace)
        online=JointTrajectoryObserver(Reference(),V2)
        first=None
        for i,q in enumerate(trace):
            a=online.observe(type('S',(),{'joint_position':q})(),i*.02)
            if first is None and a['failure']:first=a['failure']['first_violation_time_s']
        online=online.finalize()
        self.assertEqual(offline,online)
        self.assertEqual(first,offline['failure']['first_violation_time_s'])

    def test_runtime_gate_and_local_permit(self):
        adapter=BFMAdapter(device='cpu')
        adapter.reference=Reference()
        adapter.capabilities.require_envelope=True
        adapter.capabilities.withhold('TRACK_MOTION','semantic validation pending')
        with tempfile.TemporaryDirectory() as directory:
            runtime=ExecutionRuntime(adapter,PHYSICAL,directory,track_v2_config=V2)
            for instruction in (Instruction('TRACK_MOTION','dance_prefix_500'),
                                Instruction('TRACK_MOTION','dance_prefix_500',{'contract_profile':'JOINT_TRAJECTORY_V2'}),
                                Instruction('TRACK_MOTION','alias',{'contract_profile':'JOINT_TRAJECTORY_V2'})):
                with self.assertRaises(RuntimeError):runtime.start_instruction(instruction)
            with self.assertRaisesRegex(RuntimeError,'EVALUATION_PERMIT_REQUIRED'):
                runtime.start_instruction(Instruction('STAND'),True)
            self.assertIsNone(runtime.active)

    def test_v2_envelope_scope_and_legacy_noninheritance(self):
        registry=CapabilityRegistry();registry.require_envelope=True
        registry.add('TRACK_MOTION',['API_AVAILABLE'])
        base=dict(behavior='TRACK_MOTION',target_id='dance_prefix_500',asset_sha256='asset',
            checkpoint_sha256='model',simulator_version='3.3.7',adapter_version='adapter',
            contract_hash='v2',scope='SIM_ONLY',candidate_region={},empirical_support=(),
            runtime_entry_conditions={},initialization='reference_first',previous_behavior=None,
            policy_history='RESET',reference_alignment='FIXED_WORLD_V1',split='D2',
            counts={'contract_success':1},failure_breakdown={},terminal_errors={},
            completion_times_s={},quality_metrics={},evidence_paths=('evidence',))
        identity=dict(checkpoint_sha256='model',mujoco='3.3.7',adapter_version='adapter',
                      reference_preprocessing='FIXED_WORLD_V1')
        registry.record_envelope(ValidationEnvelope(**base))
        self.assertEqual(registry.describe_capabilities(),[])
        with self.assertRaisesRegex(RuntimeError,'NO_MATCHING_VALIDATION_ENVELOPE'):
            registry.authorize('TRACK_MOTION','v2','dance_prefix_500',identity,'asset','JOINT_TRAJECTORY_V2')
        registry.record_envelope(ValidationEnvelope(**{**base,'semantic_profile':'JOINT_TRAJECTORY_V2'}))
        registry.authorize('TRACK_MOTION','v2','dance_prefix_500',identity,'asset','JOINT_TRAJECTORY_V2')
        for changed in (dict(checkpoint_sha256='other'),dict(mujoco='other'),
                        dict(adapter_version='other'),dict(reference_preprocessing='other')):
            with self.assertRaisesRegex(RuntimeError,'NO_MATCHING_VALIDATION_ENVELOPE'):
                registry.authorize('TRACK_MOTION','v2','dance_prefix_500',{**identity,**changed},'asset','JOINT_TRAJECTORY_V2')
        for contract,target,asset in (('other','dance_prefix_500','asset'),
                                      ('v2','alias','asset'),('v2','dance_prefix_500','other')):
            with self.assertRaisesRegex(RuntimeError,'NO_MATCHING_VALIDATION_ENVELOPE'):
                registry.authorize('TRACK_MOTION',contract,target,identity,asset,'JOINT_TRAJECTORY_V2')

    def test_manifest_is_independent_and_fixed(self):
        from evaluation.phase31 import manifest
        phase3=json.loads((ROOT/'tests/fixtures/phase3_config.json').read_text())
        a=manifest(phase3);b=manifest(phase3)
        self.assertEqual(a,b);self.assertEqual(len(a),64)
        self.assertEqual(len({t['trial_id'] for t in a}),64)
        self.assertEqual(a[0]['proposal'],a[40]['proposal'])

    def test_handoff_manifest_detects_missing_duplicate_unexpected(self):
        from evaluation.phase31 import audit_episode
        trial=dict(trial_id='synthetic',behavior='CHAIN',sequence_length=1)
        base=['STAND_INITIAL','TRACK_1','STAND_EXPLICIT_1']
        def audit(names):
            row=dict(status='SUCCESS',stages=[dict(stage=n,target_behavior='STAND') for n in names],
                     final_guard_calls={'blocked_reset':0,'blocked_set_state':0})
            return audit_episode(trial,row)
        self.assertEqual(audit(base)['missing_handoffs'],[])
        self.assertTrue(audit(base[:2])['missing_handoffs'])
        self.assertTrue(audit(base+base[1:])['duplicate_handoffs'])
        self.assertTrue(audit([base[0],base[2],base[1]])['unexpected_handoffs'])

    def test_semantic_failure_latches_through_successful_exit(self):
        class Fake:
            mass=30
            identity={'scope':'UNIT_TEST'}
            def __init__(self):
                self.reference=Reference()
                self.s=StateSnapshot(0,0,(0,0,.8),(1,0,0,0),0,(0,0,0),(0,0,0),
                    tuple(np.zeros(29)),tuple(np.zeros(29)),.8,0,(),400,200,False,())
                self.capabilities=CapabilityRegistry()
            def create_observer(self,config):return JointTrajectoryObserver(self.reference,config)
            def snapshot(self):return self.s
            def prepare(self,instruction):return BodyObjective('TRACK','test',{}),None,{}
            def step(self,elapsed,tracking=False):self.s=replace(self.s,sim_time=self.s.sim_time+.02)
            def use_stand(self):pass
            def tracking_metrics(self,s,t):return {}
        with tempfile.TemporaryDirectory() as directory:
            runtime=ExecutionRuntime(Fake(),PHYSICAL,directory,register_validation=False,
                evaluation_permit=issue_local_evaluation_permit('latch_test'),track_v2_config=V2)
            runtime.start_instruction(Instruction('TRACK_MOTION','dance_prefix_500',
                {'contract_profile':'JOINT_TRAJECTORY_V2'}),True)
            result=None
            for _ in range(750):
                result=runtime.tick()
                if result:break
            self.assertIsNotNone(result)
            self.assertEqual(result.status,'FAILED')
            self.assertEqual(result.semantic_assessment['status'],'FAIL')
            self.assertEqual(result.exit_stabilization_outcome,'SUCCESS')


if __name__=='__main__':unittest.main()
