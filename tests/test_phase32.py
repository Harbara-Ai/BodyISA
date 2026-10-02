"""Synthetic admission tests and real-simulator read/rejection tests; not policy success evidence."""
from pathlib import Path
from dataclasses import replace
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from bodyisa.core import ExecutionRuntime,ExecutionContract,Instruction,StateSnapshot,issue_local_evaluation_permit
from bodyisa.admission import EntryPolicy,SourceReceipt
from bodyisa.evidence import verify_bundle,file_hash,digest,canonical,IDENTITY_KEYS
from bodyisa.bfm_adapter import BFMAdapter

ROOT=Path(__file__).resolve().parents[1]
POLICY=json.loads((ROOT/'bodyisa/profiles/g1/entry_policy.json').read_text())
PHYSICAL=json.loads((ROOT/'bodyisa/profiles/g1/physical.json').read_text())
V2=json.loads((ROOT/'bodyisa/profiles/g1/track_v2.json').read_text())


def fixture(root,opcode='TRACK_MOTION',name='track'):
    identity={key:'test_'+key for key in IDENTITY_KEYS};identity['entry_policy_hash']=digest(POLICY)
    identity['capabilities']=[dict(opcode='STAND',target_id=None,semantic_profile='STAND_REGION_V1',scope='SIM_ONLY'),dict(opcode='TRACK_MOTION',target_id='dance_prefix_500',semantic_profile='JOINT_TRAJECTORY_V2',scope='SIM_ONLY')]
    identity.update(physical_contract_hash=digest(PHYSICAL),semantic_contract_hash=digest(V2))
    cap=dict(opcode=opcode,target_id='dance_prefix_500' if opcode=='TRACK_MOTION' else None,
             semantic_profile='JOINT_TRAJECTORY_V2' if opcode=='TRACK_MOTION' else 'STAND_REGION_V1',scope='SIM_ONLY')
    refs=[]
    for role in ('stage_a_results','stage_a_manifest','frozen_config','entry_policy','compatibility','regression','execution_log'):
        path=root/(name+'_'+role+'.txt');path.write_text('local unit fixture '+role)
        refs.append(dict(role=role,path=path.name,sha256=file_hash(path)))
    review=root/(name+'_review.json')
    review.write_text(json.dumps(dict(status='APPROVED',review_id='unit_review',reviewer='LOCAL_AUTOMATED_REVIEW',
        reason='unit fixture',identity_hash=digest(identity),entry_policy_hash=digest(POLICY),criteria={'unit':True})))
    bundle=dict(schema='BODYISA_EVIDENCE_V1',capability=cap,identity=identity,
        entry_policy_hash=digest(POLICY),payloads=refs,review=dict(path=review.name,sha256=file_hash(review)),
        validation=dict(scope='SIM_ONLY',requested=1,accepted=1,success=1,failures=0,rejections=0,source_run='synthetic_unit'))
    path=root/(name+'_bundle.json');path.write_text(json.dumps(bundle))
    anchor=root/(name+'_anchor.json')
    def pin():
        anchor.write_text(json.dumps(dict(schema='BODYISA_LOCAL_TRUST_V1',approved_bundles={file_hash(path):
            dict(status='APPROVED',path=path.name,review_id='unit_review',review_sha256=file_hash(review))})))
    pin()
    return path,anchor,identity,bundle,pin


def stand_snapshot():
    return StateSnapshot(**json.loads((ROOT/'tests/fixtures/stand_snapshot.json').read_text()))



class Phase32EvidenceTests(unittest.TestCase):
    def test_experiment_tools_compile_before_freezing(self):
        for path in (ROOT/'evaluation').glob('*.py'):
            compile(path.read_text(encoding='utf-8'),str(path),'exec')

    def test_verified_result_is_immutable_and_tamper_cannot_self_certify(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,anchor,identity,bundle,pin=fixture(root)
            verified=verify_bundle(path,root,anchor,identity)
            copied=verified.payload();copied['capability']['scope']='REAL_ROBOT'
            self.assertEqual(verified.payload()['capability']['scope'],'SIM_ONLY')
            payload=root/bundle['payloads'][0]['path'];payload.write_text('tampered')
            with self.assertRaisesRegex(RuntimeError,'PAYLOAD_HASH'):verify_bundle(path,root,anchor,identity)
            bundle['payloads'][0]['sha256']=file_hash(payload);path.write_text(json.dumps(bundle))
            with self.assertRaisesRegex(RuntimeError,'NO_TRUSTED_REVIEW'):verify_bundle(path,root,anchor,identity)

    def test_missing_evidence_outside_root_and_unknown_anchor(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,anchor,identity,bundle,pin=fixture(root)
            (root/bundle['payloads'][-1]['path']).unlink()
            with self.assertRaisesRegex(RuntimeError,'REQUIRED_EVIDENCE_MISSING'):verify_bundle(path,root,anchor,identity)
            with self.assertRaisesRegex(RuntimeError,'OUTSIDE_ALLOWED_ROOT'):verify_bundle(root.parent/'outside.json',root,anchor,identity)
            bundle.pop('identity');path.write_text(json.dumps(bundle));pin()
            with self.assertRaisesRegex(RuntimeError,'REQUIRED_FIELD'):verify_bundle(path,root,anchor,identity)

    def test_legacy_target_profile_backend_contract_scope(self):
        for field,value in [('target_id','alias'),('semantic_profile','LEGACY_PROCESS'),('scope','REAL_ROBOT')]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);path,anchor,identity,bundle,pin=fixture(root)
                bundle['capability'][field]=value;path.write_text(json.dumps(bundle));pin()
                with self.assertRaisesRegex(RuntimeError,'PROFILE_TARGET_SCOPE'):verify_bundle(path,root,anchor,identity)
        for key in ('checkpoint_sha256','semantic_contract_hash','asset_sha256','simulator_version','source_sha256','reference_preprocessing','controller_parameters_sha256'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);path,anchor,identity,bundle,pin=fixture(root)
                with self.assertRaisesRegex(RuntimeError,'BACKEND_CONTRACT_SCOPE'):
                    verify_bundle(path,root,anchor,{**identity,key:'different'})

    def test_revoked_or_unreviewed_anchor(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,anchor,identity,bundle,pin=fixture(root)
            record=json.loads(anchor.read_text());next(iter(record['approved_bundles'].values()))['status']='REVOKED'
            anchor.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError,'WITHHELD_OR_REVOKED'):verify_bundle(path,root,anchor,identity)
            record['approved_bundles']={};anchor.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError,'NO_TRUSTED_REVIEW'):verify_bundle(path,root,anchor,identity)

    def test_atomic_import_preserves_independent_stand_and_idle_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);sp,sa,identity,_,_=fixture(root,'STAND','stand')
            tp,ta,_,bundle,_=fixture(root)
            adapter=BFMAdapter(device='cpu')
            runtime=ExecutionRuntime(adapter,PHYSICAL,root/'logs',track_v2_config=V2,entry_policy=POLICY,
                allowed_evidence_root=root,trusted_registry_path=sa)
            runtime.initialize_episode('unit')
            before=adapter.context_identity()
            with patch.object(adapter,'evidence_identity',return_value=identity):runtime.load_evidence(sp)
            self.assertEqual(before,adapter.context_identity())
            self.assertEqual([x['opcode'] for x in runtime.describe_capabilities()],['STAND'])
            with self.assertRaisesRegex(RuntimeError,'NO_TRUSTED_EVIDENCE'):
                runtime.start_instruction(Instruction('TRACK_MOTION','dance_prefix_500',{'contract_profile':'JOINT_TRAJECTORY_V2'}))
            self.assertEqual(before,adapter.context_identity())
            (root/bundle['payloads'][0]['path']).unlink();runtime.trusted_registry_path=ta
            with patch.object(adapter,'evidence_identity',return_value=identity),self.assertRaises(RuntimeError):runtime.load_evidence(tp)
            self.assertEqual([x['opcode'] for x in runtime.describe_capabilities()],['STAND'])
            self.assertEqual(before,adapter.context_identity())
            runtime.active={'instruction':Instruction('STAND')}
            with self.assertRaisesRegex(RuntimeError,'BUSY'):runtime.load_evidence(sp)


class Phase32EntryTests(unittest.TestCase):
    def test_standing_outside_motion_region_is_distinct_from_invariant(self):
        s=stand_snapshot();changed=replace(s,joint_position=tuple(np.asarray(s.joint_position)+.03))
        predicates,_,violations,_=ExecutionContract(PHYSICAL,35).evaluate(changed,0,6)
        self.assertFalse(violations);self.assertTrue(all(predicates.values()))
        region=EntryPolicy(POLICY).region('STAND_TERMINAL',changed)
        self.assertFalse(region['eligible']);self.assertFalse(region['predicates']['joint_rmse_rad'])

    def test_region_features_and_circular_heading(self):
        policy=EntryPolicy(POLICY);s=stand_snapshot()
        self.assertTrue(policy.region('STAND_TERMINAL',s)['eligible'])
        variations=[replace(s,joint_velocity=tuple(np.ones(29))),replace(s,base_linear_velocity_world=(1,0,0)),
            replace(s,base_angular_velocity_world=(0,0,1)),replace(s,tilt=.2),replace(s,root_height=.9),
            replace(s,base_position_world=(1,1,.8)),replace(s,base_heading_world=1.)]
        for snapshot in variations:self.assertFalse(policy.region('STAND_TERMINAL',snapshot)['eligible'])
        cfg=policy.config;cfg['regions']['STAND_TERMINAL']['center']['base_heading_world']=np.pi-.01
        self.assertTrue(EntryPolicy(cfg).region('STAND_TERMINAL',replace(s,base_heading_world=-np.pi+.01))['predicates']['heading_distance_rad'])

    def test_receipt_context_revision_history_and_repeat(self):
        context=dict(reset_generation=1,control_revision=50,history_sha256='history',last_action_sha256='last')
        receipt=SourceReceipt('session','episode','STAND','contract','SUCCESS','source',1,canonical(context),2)
        policy=EntryPolicy(POLICY);s=stand_snapshot()
        self.assertTrue(policy.check('TRACK_MOTION',s,'STAND_DONE',receipt,context,'session','episode',2,None)['eligible'])
        for changed in ({**context,'reset_generation':2},{**context,'history_sha256':'cleared'},{**context,'control_revision':51}):
            self.assertFalse(policy.check('TRACK_MOTION',s,'STAND_DONE',receipt,changed,'session','episode',2,None)['eligible'])
        self.assertFalse(policy.check('TRACK_MOTION',s,'STAND_DONE',receipt,context,'session','other',2,None)['eligible'])
        self.assertFalse(policy.check('TRACK_MOTION',s,'STAND_DONE',receipt,context,'session','episode',3,None)['eligible'])
        self.assertFalse(policy.check('TRACK_MOTION',s,'CLOSED',receipt,context,'session','episode',2,None)['eligible'])

    def test_real_simulator_queries_rejections_and_context_change_are_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);adapter=BFMAdapter(device='cpu')
            runtime=ExecutionRuntime(adapter,PHYSICAL,root,track_v2_config=V2,entry_policy=POLICY,
                evaluation_permit=issue_local_evaluation_permit('synthetic_context_test'))
            runtime.initialize_episode('unit')
            before=adapter.context_identity()
            for _ in range(3):runtime.get_state();runtime.get_execution();runtime.describe_capabilities()
            self.assertEqual(before,adapter.context_identity())
            for instruction in (Instruction('STAND'),Instruction('TRACK_MOTION','alias',{'contract_profile':'JOINT_TRAJECTORY_V2'}),
                Instruction('STAND',parameters={'trusted':True}),Instruction('TRACK_MOTION','dance_prefix_500',{'evidence_path':'arbitrary'})):
                with self.assertRaises(RuntimeError):runtime.start_instruction(instruction)
                self.assertEqual(before,adapter.context_identity());self.assertIsNone(runtime.active)
            # Pure context fixture, not a BFM rollout or ordinary API success claim.
            adapter.snapshot=lambda:stand_snapshot()
            runtime._entry_phase='STAND_DONE'
            runtime._source_receipt=SourceReceipt(runtime.session_id,runtime.episode_id,'STAND',runtime.config_hash,
                'SUCCESS','synthetic_receipt',stand_snapshot().sim_time,canonical(before),0)
            request=Instruction('TRACK_MOTION','dance_prefix_500',{'contract_profile':'JOINT_TRAJECTORY_V2'})
            self.assertTrue(runtime.preflight(request,True)['eligible'])
            adapter.control_revision+=1
            changed=adapter.context_identity()
            with self.assertRaisesRegex(RuntimeError,'SOURCE_STATE_OR_HISTORY_CHANGED'):runtime.start_instruction(request,True)
            self.assertEqual(changed,adapter.context_identity());self.assertIsNone(runtime.active)
            runtime.active={'instruction':Instruction('STAND')}
            with self.assertRaisesRegex(RuntimeError,'BUSY'):runtime.start_instruction(request,True)
            self.assertEqual(changed,adapter.context_identity())

    def test_output_unavailable_does_not_change_decision(self):
        from bodyisa.track_contract_v2 import JointTrajectoryObserver
        ref=type('R',(),{'joints':np.tile(np.linspace(0,1,500)[:,None],(1,29))})()
        observer=JointTrajectoryObserver(ref,V2)
        result=observer.observe(type('S',(),{'joint_position':ref.joints[0]})(),0.)
        self.assertEqual(result['status'],'PENDING')
        self.assertEqual(result['metrics']['rolling_window'],dict(value=None,available=False,reason='INSUFFICIENT_SAMPLES'))

    def test_real_adapter_reset_invalidates_initialization_without_starting_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter=BFMAdapter(device='cpu')
            runtime=ExecutionRuntime(adapter,PHYSICAL,temp,entry_policy=POLICY,track_v2_config=V2,
                evaluation_permit=issue_local_evaluation_permit('reset_fixture'))
            runtime.initialize_episode('reset_test')
            generation=adapter.reset_generation
            adapter.obs=adapter.env.reset()  # separate local regression scene, outside formal A/B
            self.assertGreater(adapter.reset_generation,generation)
            before=adapter.context_identity()
            with self.assertRaisesRegex(RuntimeError,'INITIALIZATION_CONTEXT_CHANGED'):
                runtime.start_instruction(Instruction('STAND'),True)
            self.assertEqual(before,adapter.context_identity());self.assertIsNone(runtime.active)


if __name__=='__main__':unittest.main()
