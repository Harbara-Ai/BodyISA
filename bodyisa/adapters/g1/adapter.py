"""Boundary to the unchanged minimal BFM implementation and MuJoCo."""
from pathlib import Path
import sys
import time
import hashlib
import numpy as np
import mujoco
import torch
from scipy.spatial.transform import Rotation, Slerp
from bodyisa.core import StateSnapshot, ContactSnapshot, BodyObjective, CapabilityRegistry, PoseRegistry

from .config import external_root
MINIMAL = external_root()
sys.path.insert(0,str(MINIMAL))
from common import (DEFAULT_JOINT_POS, POLICY_JOINT_NAMES, ACTION_RESCALE,
                    ACTION_SCALES, KP_GAINS, KD_GAINS)
from env import MuJoCoBFMZeroEnv, calc_angular_velocity


def make_env():
    return MuJoCoBFMZeroEnv(str(MINIMAL/'bfm_zero_inference_code/g1_for_reward_inference.xml'),
        kp_gains=KP_GAINS,kd_gains=KD_GAINS,default_joint_pos=DEFAULT_JOINT_POS,
        action_scales=ACTION_SCALES,action_rescale=ACTION_RESCALE,enable_video=False)


class ReferenceMotion:
    def __init__(self, path, frames=500):
        self.path = str(Path(path).resolve())
        with np.load(path) as f:
            self.fps = float(np.asarray(f['fps']).item())
            names = list(f['dof_names'].astype(str))
            if names != POLICY_JOINT_NAMES: raise ValueError('Reference joint order mismatch')
            self.joints = f['dof_positions'][:frames].copy()
            self.positions = f['body_positions'][:frames].copy()
            self.quaternions = f['body_rotations'][:frames].copy()
        if self.fps <= 0 or len(self.joints)<2: raise ValueError('Invalid reference timing')
        self.times = np.arange(len(self.joints))/self.fps
        self.duration = float(self.times[-1])
        self.slerps = [Slerp(self.times, Rotation.from_quat(self.quaternions[:,i,[1,2,3,0]]))
                       for i in range(self.quaternions.shape[1])]
        self.reference_id = f'dance_prefix_{len(self.joints)}'
        self.source_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.body_position_source = ('PLACEHOLDER_ZERO_NONROOT' if
            np.all(self.positions[:,1:]==0) else 'UNNAMED_BODY_POSITIONS')

    def sample(self, t):
        t = float(np.clip(t,0,self.duration))
        idx = min(int(np.searchsorted(self.times,t,side='right'))-1,len(self.times)-2)
        alpha = (t-self.times[idx])/(self.times[idx+1]-self.times[idx])
        joints = self.joints[idx]*(1-alpha)+self.joints[idx+1]*alpha
        pos = self.positions[idx]*(1-alpha)+self.positions[idx+1]*alpha
        quat = np.stack([s(t).as_quat()[[3,0,1,2]] for s in self.slerps])
        return joints,pos,quat


class G1Adapter:
    def __init__(self, model=None, device='cuda', env=None, paths=None):
        self.paths = paths or {}
        self.model, self.device = model,device
        self.env = env if env is not None else make_env()
        self.obs = self.env.reset()
        m = self.env.mjm
        names = [mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(1,m.njnt)]
        if names != POLICY_JOINT_NAMES: raise ValueError('Simulator joint order mismatch')
        self.dt = self.env.sim_dt*self.env.n_substeps
        self.mass = float(m.body_mass.sum())
        self.scratch = mujoco.MjData(m)
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state_buffer = np.empty(mujoco.mj_stateSize(m,self.state_spec))
        self.body_ids = []
        head = None
        self.foot_bodies = {}
        for i in range(m.nbody):
            name = mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) or ''
            if name=='head_link': head=i
            elif not(name.startswith('world') or name.startswith('dummy') or name.endswith('hand')):
                self.body_ids.append(i)
            ancestor=i
            while ancestor:
                ancestor_name = mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,ancestor)
                if ancestor_name in ('left_ankle_roll_link','right_ankle_roll_link'):
                    self.foot_bodies[i]=ancestor_name
                    break
                ancestor=int(m.body_parentid[ancestor])
        if head is not None: self.body_ids.append(head)
        self.floor = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'floor')
        if self.floor < 0: raise ValueError('Flat floor geom unavailable')
        self.poses = PoseRegistry()
        self.capabilities = CapabilityRegistry()
        for name in ('goal_inference','tracking_inference','reward_inference'):
            self.capabilities.add(name,['API_AVAILABLE'],implementation='BACKEND_API')
        for name in ('STAND','REACH_POSE','TRACK_MOTION'):
            self.capabilities.add(name,['API_AVAILABLE'])
        for name in ('GO_TO','TURN_TO','REGULATE'):
            self.capabilities.add(name,[],implementation='NOT_IMPLEMENTED')
        self.reference = None
        self.track_z = None
        self.current_z = None
        self.identity = dict(embodiment='G1',controller='BFM-Zero',execution_backend='MuJoCo',
            adapter='minimal_inference', model='external_checkpoint',
            adapter_version='phase3-boundary-v1',
            reference_preprocessing='FIXED_WORLD_V1',
            checkpoint_sha256='d8d07d7b83ad030286f44d55a48899dc42db4b48000132fffad063653a6eb49e',
            mujoco=mujoco.__version__,torch=torch.__version__,device=device,
            control_dt_s=self.dt,joint_order=POLICY_JOINT_NAMES,quaternion_order='wxyz')
        self.reset_generation=0
        self.control_revision=0
        for method_name in ('reset','set_state','step'):
            original=getattr(self.env,method_name)
            def tracked(*args,_original=original,_name=method_name,**kwargs):
                self.control_revision+=1
                if _name in ('reset','set_state'):self.reset_generation+=1
                return _original(*args,**kwargs)
            setattr(self.env,method_name,tracked)

    def context_identity(self):
        from bodyisa.runtime.evidence import array_hash,digest
        state=np.empty(mujoco.mj_stateSize(self.env.mjm,self.state_spec))
        mujoco.mj_getState(self.env.mjm,self.env.mjd,state,self.state_spec)
        history={}
        for key in sorted(vars(self.env)):
            if key.startswith('history') or key in ('last_action','body_pos_prev','body_quat_prev','step_count'):
                value=getattr(self.env,key)
                history[key]=None if value is None else array_hash(value)
        obs={key:array_hash(value) for key,value in self.obs.items()}
        return dict(reset_generation=self.reset_generation,control_revision=self.control_revision,
            sim_time=float(self.env.mjd.time),integration_sha256=array_hash(state),
            history_sha256=digest(history),last_action_sha256=history.get('last_action'),
            cached_observation_sha256=digest(obs),current_latent_sha256=None if self.current_z is None else array_hash(self.current_z))

    def snapshot(self):
        m,d = self.env.mjm,self.env.mjd
        # Only scratch is forwarded; live simulator/history/control are never written.
        mujoco.mj_getState(m,d,self.state_buffer,self.state_spec)
        mujoco.mj_setState(m,self.scratch,self.state_buffer,self.state_spec)
        mujoco.mj_forward(m,self.scratch)
        x = self.scratch
        q = x.qpos[3:7].copy()
        r = Rotation.from_quat(q[[1,2,3,0]])
        w,a,b,c = q
        yaw = float(np.arctan2(2*(w*c+a*b),1-2*(b*b+c*c)))
        tilt = float(np.arccos(np.clip(r.as_matrix()[2,2],-1,1)))
        contacts=[]
        foot_forces={}
        nonfoot=False
        for i in range(x.ncon):
            contact=x.contact[i]
            if self.floor not in (contact.geom1,contact.geom2): continue
            other=int(contact.geom2 if contact.geom1==self.floor else contact.geom1)
            body=int(m.geom_bodyid[other])
            foot=self.foot_bodies.get(body)
            force=np.zeros(6)
            mujoco.mj_contactForce(m,x,i,force)
            normal=float(max(0,force[0]))
            active=bool(contact.dist<=0 and contact.efc_address>=0)
            if active and foot: foot_forces[foot]=foot_forces.get(foot,0)+normal
            if active and not foot: nonfoot=True
            contacts.append(ContactSnapshot(other,body,foot,float(contact.dist),normal,active))
        return StateSnapshot(float(d.time),time.time(),tuple(x.qpos[:3]),tuple(q),yaw,
            tuple(x.qvel[:3]),tuple(r.apply(x.qvel[3:6])),tuple(x.qpos[7:]),
            tuple(x.qvel[6:]),float(x.qpos[2]),tilt,tuple(contacts),
            float(sum(foot_forces.values())),float(max(foot_forces.values(),default=0)),
            nonfoot,tuple(tuple(row) for row in x.xpos[self.body_ids]))

    @torch.inference_mode()
    def encode_default_pose(self, cache_dir):
        if self.model is None: raise RuntimeError('BFM model required')
        reference_env=make_env()
        reference_env.reset()
        reference_env.set_state(DEFAULT_JOINT_POS.copy())
        obs=reference_env._create_observation_backward()
        z=self.model.goal_inference({k:v.to(self.device) for k,v in obs.items()})
        if not torch.isfinite(z).all(): raise ValueError('Nonfinite goal latent')
        path=Path(cache_dir)/'default_pose_latent.npz'
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,z=z.cpu().numpy(),joint_target=DEFAULT_JOINT_POS)
        self.poses.register('default_pose','minimal_inference/common.py:DEFAULT_JOINT_POS',
                            DEFAULT_JOINT_POS,str(path.resolve()),z)
        for name in ('STAND','REACH_POSE'):
            self.capabilities.entries[name]['evidence_levels'].append('ASSET_AVAILABLE')

    @torch.inference_mode()
    def encode_reference(self, reference, cache_dir):
        if self.model is None: raise RuntimeError('BFM model required')
        self.reference=reference
        e=make_env()
        self.metric_env=make_env()
        e.reset()
        # Existing privileged builder differences divide by policy dt: resample FIRST.
        times=np.arange(int(np.ceil(reference.duration/self.dt))+1)*self.dt
        seq=[]
        previous=None
        for t in times:
            joints,pos,quat=reference.sample(t)
            if previous is None:
                jv,rv,av=np.zeros(29),np.zeros(3),np.zeros(3)
            else:
                pj,pp,pq=previous
                jv=(joints-pj)/self.dt
                rv=(pos[0]-pp[0])/self.dt
                av=calc_angular_velocity(quat[0],pq[0],self.dt)
            e.set_state(joints,jv,root_quat=quat[0],root_pos=pos[0],root_vel=rv,root_ang_vel=av)
            seq.append(e._create_observation_backward())
            previous=joints,pos,quat
        obs={k:torch.cat([v[k] for v in seq]).to(self.device) for k in seq[0]}
        self.track_z=self.model.tracking_inference(obs)
        if not torch.isfinite(self.track_z).all(): raise ValueError('Nonfinite tracking latent')
        path=Path(cache_dir)/'dance_prefix_latents.npz'
        np.savez(path,z=self.track_z.cpu().numpy(),times=times,source_fps=reference.fps)
        self.track_source=dict(mechanism='tracking_inference',cache=str(path.resolve()),
            reference_path=reference.path,reference_sha256=reference.source_hash,
            reference_id=reference.reference_id,source_fps=reference.fps,
            frames=len(reference.joints),duration_s=reference.duration,latent_period_s=self.dt,
            body_metric_source='FK_DERIVED_FROM_REFERENCE_JOINTS_AND_ROOT',
            asset_body_position_status=reference.body_position_source,
            body_order='minimal get_privileged_state filtered body order including head')
        self.capabilities.entries['TRACK_MOTION']['evidence_levels'].append('ASSET_AVAILABLE')

    def initialize_reference(self):
        j,p,q=self.reference.sample(0)
        self.env.set_state(j,root_pos=p[0],root_quat=q[0])
        # Bootstrap only; policy observation creation is a control operation, never a read.
        self.obs=self.env._create_observation()

    def use_stand(self):
        self.current_z=self.poses.get('default_pose')['latent']

    def prepare(self, instruction):
        if instruction.parameters and not (instruction.opcode=='TRACK_MOTION' and
            instruction.parameters=={'contract_profile':'JOINT_TRAJECTORY_V2'}):
            raise ValueError('Only registered targets and explicit v2 profile are accepted')
        if instruction.opcode=='TRACK_MOTION':
            if self.reference is None or instruction.target_id!=self.reference.reference_id:
                raise ValueError('Unregistered reference')
            self.current_z=self.track_z[:1]
            return BodyObjective('TRACK','registered_motion',dict(reference_id=instruction.target_id)),None,self.track_source
        pose_id='default_pose' if instruction.opcode=='STAND' else instruction.target_id
        if instruction.opcode=='STAND' and instruction.target_id is not None:
            raise ValueError('STAND has no target ID')
        entry=self.poses.get(pose_id)
        self.current_z=entry['latent']
        return (BodyObjective('REACH','standing_region' if instruction.opcode=='STAND' else 'joint_pose',
                             dict(pose_id=pose_id)),entry['joint_target'],
                dict(mechanism='goal_inference',pose_id=pose_id,cache=entry['latent_reference'],source=entry['source']))

    @torch.inference_mode()
    def step(self, elapsed, tracking=False):
        z=self.current_z
        if tracking:
            index=min(int(np.floor((elapsed+1e-9)/self.dt)),len(self.track_z)-1)
            z=self.track_z[index:index+1]
        action=self.model.act({k:v.unsqueeze(0).to(self.device) for k,v in self.obs.items()},z,mean=True)[0].cpu().numpy()
        if not np.isfinite(action).all(): raise ValueError('NONFINITE_ACTION')
        _,self.obs,_,_=self.env.step(action)

    def tracking_metrics(self,s,t):
        j,p,q=self.reference.sample(t)
        body=np.asarray(s.body_position_world)
        # Supplied non-root asset points are zero placeholders, not world positions.
        # Derive an explicitly labelled FK reference using the same robot morphology.
        self.metric_env.set_state(j,root_pos=p[0],root_quat=q[0])
        target_body=self.metric_env.mjd.xpos[self.body_ids].copy()
        return dict(joint_tracking_rmse=float(np.sqrt(np.mean((np.asarray(s.joint_position)-j)**2))),
                    root_tracking_error=float(np.linalg.norm(np.asarray(s.base_position_world)-p[0])),
                    body_tracking_rmse=float(np.sqrt(np.mean((body-target_body)**2))))

    @property
    def supported_instructions(self):
        from bodyisa.core import Instruction
        return (Instruction('STAND'),Instruction('TRACK_MOTION','dance_prefix_500',{'contract_profile':'JOINT_TRAJECTORY_V2'}))

    @property
    def capability_catalog(self):
        return [dict(opcode='STAND',target_id=None,semantic_profile='STAND_REGION_V1',scope='SIM_ONLY'),
                dict(opcode='TRACK_MOTION',target_id='dance_prefix_500',semantic_profile='JOINT_TRAJECTORY_V2',scope='SIM_ONLY')]

    def create_observer(self,config):
        from bodyisa.profiles.g1.track_contract_v2 import JointTrajectoryObserver
        return JointTrajectoryObserver(self.reference,config)

    def evidence_identity(self,physical,v2,entry_policy):
        from .identity import actual_identity
        return actual_identity(self,physical,v2,entry_policy)

    def initialize_nominal(self,joint_delta,max_delta):
        delta=np.zeros(len(POLICY_JOINT_NAMES)) if joint_delta is None else np.asarray(joint_delta,float)
        if delta.shape!=(len(POLICY_JOINT_NAMES),) or not np.isfinite(delta).all() or np.max(np.abs(delta))>max_delta:
            raise RuntimeError('INITIALIZATION_PROPOSAL_OUTSIDE_SCOPE')
        self.obs=self.env.reset()
        if np.any(delta):
            s=self.snapshot()
            self.env.set_state(np.asarray(s.joint_position)+delta,
                root_pos=np.asarray(s.base_position_world),root_quat=np.asarray(s.base_orientation_world))
            self.obs=self.env._create_observation()
        return delta

    def close(self):
        for env in (self.env,getattr(self,'metric_env',None)):
            close=getattr(env,'close',None)
            if close:close()
