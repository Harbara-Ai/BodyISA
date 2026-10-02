import unittest
import json
import tempfile
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np
import mujoco
from bodyisa.core import (ExecutionContract, CapabilityRegistry, ExecutionRuntime,
                         Instruction, TransitionContract, BodyObjective)
from bodyisa.bfm_adapter import BFMAdapter, ReferenceMotion

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'bodyisa/profiles/g1/physical.json').read_text())


def integration(adapter):
    m,d=adapter.env.mjm,adapter.env.mjd
    state=np.empty(mujoco.mj_stateSize(m,adapter.state_spec))
    mujoco.mj_getState(m,d,state,adapter.state_spec)
    return state


def histories(adapter):
    e=adapter.env
    return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in vars(e).items()
            if k.startswith('history') or k in ('body_pos_prev','body_quat_prev','last_action','step_count')}


class SnapshotTests(unittest.TestCase):
    def test_reads_leave_live_state_histories_and_next_trajectory_unchanged(self):
        a,b=BFMAdapter(device='cpu'),BFMAdapter(device='cpu')
        for adapter in (a,b):
            adapter.env.step(np.ones(29,np.float32)*0.03)
            adapter.env._create_observation_backward()  # Populate reference diff history.
        before=integration(a).copy()
        history=histories(a)
        derived={k:getattr(a.env.mjd,k).copy() for k in ('xpos','xquat','qacc','ctrl','qfrc_actuator','efc_force')}
        contacts=[(int(c.geom1),int(c.geom2),float(c.dist),c.pos.copy(),c.frame.copy())
                  for c in a.env.mjd.contact]
        obs={k:v.clone() for k,v in a.obs.items()}
        for _ in range(50): a.snapshot()
        np.testing.assert_array_equal(integration(a),before)
        for k,v in history.items(): np.testing.assert_array_equal(histories(a)[k],v)
        for k,v in derived.items(): np.testing.assert_array_equal(getattr(a.env.mjd,k),v)
        self.assertEqual(len(contacts),a.env.mjd.ncon)
        for before_contact,after_contact in zip(contacts,a.env.mjd.contact):
            self.assertEqual(before_contact[:3],(int(after_contact.geom1),int(after_contact.geom2),float(after_contact.dist)))
            np.testing.assert_array_equal(before_contact[3],after_contact.pos)
            np.testing.assert_array_equal(before_contact[4],after_contact.frame)
        for k,v in obs.items(): np.testing.assert_array_equal(a.obs[k].numpy(),v.numpy())
        for i in range(8):
            action=np.sin(np.arange(29)+i).astype(np.float32)*0.03
            _,oa,_,_=a.env.step(action)
            _,ob,_,_=b.env.step(action)
            for _ in range(5): a.snapshot()
            np.testing.assert_array_equal(integration(a),integration(b))
            for k in oa: np.testing.assert_array_equal(oa[k].numpy(),ob[k].numpy())

    def test_world_angular_velocity_and_quaternion(self):
        a=BFMAdapter(device='cpu')
        a.env.set_state(np.zeros(29),root_quat=np.array([2**-.5,0,0,2**-.5]),
                        root_ang_vel=np.array([1.,0,0]))
        s=a.snapshot()
        np.testing.assert_allclose(s.base_angular_velocity_world,[1,0,0],atol=1e-7)
        self.assertAlmostEqual(s.base_heading_world,np.pi/2)
        self.assertAlmostEqual(s.base_orientation_world[0],2**-.5)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s=BFMAdapter(device='cpu').snapshot()
        cls.stable=replace(s,root_height=.8,tilt=0.,joint_position=tuple(np.zeros(29)),
            joint_velocity=tuple(np.zeros(29)),base_linear_velocity_world=(0.,0.,0.),
            base_angular_velocity_world=(0.,0.,0.),support_force=500.,foot_force_max=250.,
            nonfoot_floor_contact=False)

    def test_hold_reset_timeout_and_invariant_precedence(self):
        contract=ExecutionContract(CONFIG,30)
        for t in (0,.2):
            self.assertIsNone(contract.evaluate(replace(self.stable,sim_time=t),t,6)[-1])
        contract.evaluate(replace(self.stable,sim_time=.3,support_force=0),.3,6)
        self.assertIsNone(contract.evaluate(replace(self.stable,sim_time=.4),.4,6)[-1])
        self.assertEqual(contract.evaluate(replace(self.stable,sim_time=.9),.9,6)[-1],'SUCCESS')
        self.assertEqual(contract.evaluate(replace(self.stable,sim_time=6),6,6)[-1],'TIMEOUT')
        self.assertEqual(contract.evaluate(replace(self.stable,sim_time=6,tilt=1),6,6)[-1],'INVARIANT_VIOLATION')

    def test_pose_error_is_not_latent_elapsed_success(self):
        contract=ExecutionContract(CONFIG,30,np.ones(29)*.075)
        for t in (0,.6):
            p,e,_,reason=contract.evaluate(replace(self.stable,sim_time=t),t,6)
            self.assertFalse(p['pose_rmse'])
            self.assertAlmostEqual(e['joint_rmse'],.075)
            self.assertIsNone(reason)

    def test_height_span_resets_hold(self):
        contract=ExecutionContract(CONFIG,30)
        contract.evaluate(self.stable,0,6)
        result=contract.evaluate(replace(self.stable,sim_time=.5,root_height=.82),.5,6)
        self.assertEqual(result[1]['hold_elapsed_s'],0)
        self.assertIsNone(result[-1])

    def test_filter_and_transition_drafts(self):
        c=CapabilityRegistry()
        c.add('STAND',['API_AVAILABLE','ASSET_AVAILABLE'])
        c.add('GO_TO',[],implementation='NOT_IMPLEMENTED')
        self.assertEqual(c.describe_capabilities(),[])
        self.assertEqual(len(c.describe_capabilities(True)),2)
        for pair in [('STAND','REACH_POSE'),('REACH_POSE','STAND'),
                     ('STAND','TRACK_MOTION'),('TRACK_MOTION','STAND')]:
            self.assertEqual(TransitionContract.between(*pair,6).timeout_s,6)
        with self.assertRaises(ValueError): TransitionContract.between('REACH_POSE','TRACK_MOTION',6)

    def test_runtime_busy_cancel_unvalidated_and_drafts(self):
        a=BFMAdapter(device='cpu')
        # No model execution: stub prepare only to test ownership API.
        a.prepare=lambda i:(BodyObjective('REACH','standing_region',{}),None,{'source':'UNIT_TEST'})
        with tempfile.TemporaryDirectory() as directory:
            from bodyisa.core import issue_local_evaluation_permit
            runtime=ExecutionRuntime(a,CONFIG,directory,
                evaluation_permit=issue_local_evaluation_permit('test_phase2'))
            with self.assertRaises(RuntimeError): runtime.start_instruction(Instruction('STAND'))
            runtime.start_instruction(Instruction('STAND'),evaluation_mode=True)
            with self.assertRaises(RuntimeError): runtime.start_instruction(Instruction('STAND'),True)
            result=runtime.cancel()
            self.assertEqual(result.status,'CANCELLED')
            self.assertEqual(runtime.mode,'UNKNOWN')
            with self.assertRaises(NotImplementedError): runtime.start_instruction(Instruction('GO_TO'),True)
            self.assertEqual(runtime.describe_capabilities(),[])

    def test_reference_interpolation_30_and_60_fps(self):
        from bodyisa.bfm_adapter import POLICY_JOINT_NAMES
        with tempfile.TemporaryDirectory() as directory:
            for fps in (30,60):
                n=fps+1
                t=np.arange(n)/fps
                q=np.zeros((n,1,4)); q[:,:,0]=1
                path=Path(directory)/f'{fps}.npz'
                np.savez(path,dof_names=POLICY_JOINT_NAMES,fps=fps,
                    dof_positions=np.repeat(t[:,None],29,axis=1),
                    body_positions=np.repeat(t[:,None,None],3,axis=2),body_rotations=q)
                ref=ReferenceMotion(path,n)
                j,p,q=ref.sample(.37)
                np.testing.assert_allclose(j,.37)
                np.testing.assert_allclose(p,.37)
                self.assertAlmostEqual(ref.duration,1)
                np.testing.assert_allclose(q,[ [1,0,0,0] ])

    def test_runtime_transitions_and_tracking_requires_terminal_hold(self):
        class FakeAdapter:
            mass=30
            identity={'scope':'UNIT_TEST_ONLY'}
            def __init__(inner):
                inner.s=self.stable
                inner.capabilities=CapabilityRegistry()
                for name in ('STAND','REACH_POSE','TRACK_MOTION'):
                    inner.capabilities.add(name,['API_AVAILABLE'])
                inner.poses=type('Poses',(),{'get':lambda _,key:{'validation_status':'NOT_PHYSICS_VALIDATED'}})()
                inner.reference=type('Reference',(),{'duration':.04})()
                inner.settled=False
            def snapshot(inner): return inner.s
            def prepare(inner,i): return BodyObjective('TRACK' if i.opcode=='TRACK_MOTION' else 'REACH','test',{}),None,{'scope':'UNIT_TEST_ONLY'}
            def use_stand(inner): inner.settled=True
            def step(inner,elapsed,tracking=False): inner.s=replace(inner.s,sim_time=inner.s.sim_time+.02)
            def tracking_metrics(inner,s,t): return {'joint_tracking_rmse':0}
        with tempfile.TemporaryDirectory() as directory:
            a=FakeAdapter()
            from bodyisa.core import issue_local_evaluation_permit
            runtime=ExecutionRuntime(a,CONFIG,directory,
                evaluation_permit=issue_local_evaluation_permit('test_phase2'))
            runtime.mode='STAND'
            for opcode in ('REACH_POSE','STAND','TRACK_MOTION'):
                runtime.start_instruction(Instruction(opcode,'test' if opcode!='STAND' else None),True)
                result=None
                while result is None: result=runtime.tick()
                self.assertEqual(result.status,'SUCCESS')
                if opcode=='TRACK_MOTION':
                    self.assertTrue(a.settled)
                    self.assertGreaterEqual(result.elapsed_sim_s,.04+CONFIG['hold_s'])
                    records=[json.loads(r) for r in Path(result.evidence_path).read_text().splitlines()]
                    consume=next(r for r in records if r['event']=='REFERENCE_CONSUMED')
                    self.assertEqual(consume['transition']['source'],'TRACK_MOTION')
                    self.assertEqual(consume['transition']['target'],'STAND')
            self.assertEqual(runtime.mode,'STAND')
            # A stale successful mode cannot authorize entry from a falling state.
            a.s=replace(a.s,tilt=1)
            with self.assertRaises(RuntimeError): runtime.start_instruction(Instruction('REACH_POSE','test'),True)
            self.assertIsNone(runtime.active)


if __name__=='__main__': unittest.main()
