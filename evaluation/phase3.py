"""Finite scenario, continuous episode and scoped-evidence utilities for Phase 3."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import time
import traceback
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from bodyisa.core import ExecutionRuntime, Instruction, ValidationEnvelope
from bodyisa.adapters.g1.adapter import DEFAULT_JOINT_POS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def make_manifest(config):
    manifest=[]
    groups=[('stand_exploration','STAND','exploration',1),
        ('stand_heldout','STAND','heldout_validation',1),
        ('track_exploration','TRACK_MOTION','exploration',1),
        ('track_heldout','TRACK_MOTION','heldout_validation',1),
        ('transition_s1','CHAIN','transition',1),
        ('composition_s1','CHAIN','composition',1),
        ('composition_s3','CHAIN','composition',3),
        ('composition_s5','CHAIN','composition',5),
        ('negative_control','NEGATIVE_CONTROL','negative_control',1)]
    classes=config['scenario_distribution']['classes']
    base=config['seeds']
    for group,behavior,split,length in groups:
        for i in range(config['trials'][group]):
            seed=base[split]+i+{'STAND':0,'TRACK_MOTION':1000,'CHAIN':2000,
                                'NEGATIVE_CONTROL':3000}[behavior]+length*100
            rng=np.random.default_rng(seed)
            # Pre-generate all perturbations and signs: manifest is immutable during run.
            category=str(rng.choice(classes))
            yaw={'yaw_plus_5_deg':5,'yaw_minus_5_deg':-5,
                 'yaw_plus_10_deg':10,'yaw_minus_10_deg':-10,
                 'yaw_plus_5_joint_0.01_vel_0.02':5,
                 'yaw_minus_5_joint_0.03_vel_0.05':-5}.get(category,0)
            sigma=0.03 if 'joint_0.03' in category or category=='joint_sigma_0.03_rad' else (
                0.01 if 'joint_0.01' in category or category=='joint_sigma_0.01_rad' else 0.)
            speed=.05 if 'vel_0.05' in category or category=='horizontal_speed_0.05_mps' else (
                .02 if 'vel_0.02' in category or category=='horizontal_speed_0.02_mps' else 0.)
            direction=float(rng.uniform(-math.pi,math.pi))
            joint_delta=(rng.normal(0,sigma,29) if sigma else np.zeros(29)).tolist()
            trial_id=f'{group}_{i:03d}'
            manifest.append(dict(trial_id=trial_id,group=group,behavior=behavior,
                split=split,sequence_length=length,seed=seed,scenario_class=category,
                proposal=dict(yaw_offset_rad=math.radians(yaw),joint_delta_rad=joint_delta,
                              horizontal_velocity_world_mps=[speed*math.cos(direction),
                                                             speed*math.sin(direction),0.],
                              roll_pitch_rad=[0.,0.]),
                expected_rejection=False,initialization='reference_first' if
                behavior in ('TRACK_MOTION','NEGATIVE_CONTROL') else 'reset_default'))
    return manifest


def policy_history_fingerprint(adapter):
    e=adapter.env
    h=hashlib.sha256()
    keys=[]
    for key in sorted(vars(e)):
        if key.startswith('history') or key in ('last_action','body_pos_prev','body_quat_prev'):
            val=getattr(e,key)
            keys.append(key)
            if val is None:
                h.update(b'NONE')
                continue
            if hasattr(val,'detach'): val=val.detach().cpu().numpy()
            h.update(np.ascontiguousarray(np.asarray(val)).tobytes())
    return dict(sha256=h.hexdigest(),keys=keys,
        last_action=np.asarray(e.last_action).tolist(),step_count=int(e.step_count))


def live_state_fingerprint(adapter):
    m,d=adapter.env.mjm,adapter.env.mjd
    state=np.empty(mujoco.mj_stateSize(m,adapter.state_spec))
    mujoco.mj_getState(m,d,state,adapter.state_spec)
    return dict(sha256=hashlib.sha256(state.tobytes()).hexdigest(),sim_time=float(d.time))


def verify_handoff(source_terminal,entry_snapshot,source_history,entry_history):
    same_state=all(getattr(source_terminal,k)==getattr(entry_snapshot,k)
                   for k in source_terminal.__dataclass_fields__ if k!='wall_time')
    same_history=source_history==entry_history
    return dict(source_target_snapshot_equal_excluding_wall_time=same_state,
                source_target_sim_time_equal=source_terminal.sim_time==entry_snapshot.sim_time,
                source_target_history_equal=same_history)


class LiveMutationGuard:
    """Reject and count protected writes after one explicit episode initialization."""
    def __init__(self,adapter):
        self.adapter=adapter
        self.original_reset=adapter.env.reset
        self.original_set_state=adapter.env.set_state
        self.locked=False
        self.calls={'reset':0,'set_state':0,'blocked_reset':0,'blocked_set_state':0}
        def reset(*args,**kwargs):
            if self.locked:
                self.calls['blocked_reset']+=1
                raise RuntimeError('EPISODE_GUARD: reset forbidden')
            self.calls['reset']+=1
            return self.original_reset(*args,**kwargs)
        def set_state(*args,**kwargs):
            if self.locked:
                self.calls['blocked_set_state']+=1
                raise RuntimeError('EPISODE_GUARD: live set_state forbidden')
            self.calls['set_state']+=1
            return self.original_set_state(*args,**kwargs)
        adapter.env.reset=reset
        adapter.env.set_state=set_state

    def unlock_and_restore(self):
        self.locked=False
        self.adapter.env.reset=self.original_reset
        self.adapter.env.set_state=self.original_set_state


def initialize_episode(adapter,trial,config,guard):
    proposal=trial['proposal']
    adapter.obs=adapter.env.reset()
    if trial['initialization']=='reference_first':
        adapter.initialize_reference()
    s=adapter.snapshot()
    joint=np.asarray(s.joint_position)+np.asarray(proposal['joint_delta_rad'])
    yaw=proposal['yaw_offset_rad']
    velocity=np.asarray(proposal['horizontal_velocity_world_mps'])
    if yaw or np.any(proposal['joint_delta_rad']) or np.any(velocity):
        q=np.asarray(s.base_orientation_world)
        rotation=Rotation.from_euler('z',yaw)*Rotation.from_quat(q[[1,2,3,0]])
        root_q=rotation.as_quat()[[3,0,1,2]]
        adapter.env.set_state(joint,root_pos=np.asarray(s.base_position_world),root_quat=root_q,
            root_vel=velocity,root_ang_vel=np.asarray(s.base_angular_velocity_world))
        adapter.obs=adapter.env._create_observation()  # explicit initial actor observation
    s=adapter.snapshot()
    reasons=[]
    m=adapter.env.mjm
    for k in range(1,m.njnt):
        if m.jnt_limited[k] and not (m.jnt_range[k,0]-1e-9 <= joint[k-1] <= m.jnt_range[k,1]+1e-9):
            reasons.append(f'JOINT_LIMIT:{k}')
    if not np.isfinite([*s.base_position_world,*s.base_orientation_world,*s.joint_position,
                         *s.joint_velocity,*s.base_linear_velocity_world]).all():
        reasons.append('NONFINITE_INITIAL_STATE')
    if s.root_height<config['initialization_checks']['minimum_root_height_m']:
        reasons.append('ROOT_TOO_LOW')
    if s.tilt>config['initialization_checks']['maximum_tilt_rad']:
        reasons.append('TILT_TOO_LARGE')
    depth=config['initialization_checks']['max_floor_penetration_m']
    if any(c.distance_m < -depth for c in s.contact_snapshot): reasons.append('DEEP_FLOOR_PENETRATION')
    if s.nonfoot_floor_contact: reasons.append('NONFOOT_FLOOR_CONTACT')
    guard.locked=True
    return s,reasons


def reference_entry_mismatch(adapter,s):
    ref=adapter.reference
    j,p,q=ref.sample(0)
    r=Rotation.from_quat(np.asarray(s.base_orientation_world)[[1,2,3,0]])
    rr=Rotation.from_quat(q[0][[1,2,3,0]])
    yaw_delta=math.atan2(math.sin(s.base_heading_world-rr.as_euler('xyz')[2]),
                         math.cos(s.base_heading_world-rr.as_euler('xyz')[2]))
    return dict(joint_rmse_rad=float(np.sqrt(np.mean((np.asarray(s.joint_position)-j)**2))),
        joint_max_error_rad=float(np.max(np.abs(np.asarray(s.joint_position)-j))),
        root_position_error_m=float(np.linalg.norm(np.asarray(s.base_position_world)-p[0])),
        heading_error_rad=yaw_delta,
        linear_velocity_error_mps=float(np.linalg.norm(s.base_linear_velocity_world)),
        angular_velocity_error_radps=float(np.linalg.norm(s.base_angular_velocity_world)),
        full_orientation_error_rad=float((r.inv()*rr).magnitude()))


def stage_quality(evidence_path):
    rows=[json.loads(line) for line in Path(evidence_path).read_text(encoding='utf-8').splitlines()]
    steps=[r for r in rows if r['event']=='STEP']
    playback=[r for r in steps if r.get('reference_consumed') is False]
    metrics={}
    for key in ('joint_tracking_rmse','root_tracking_error','body_tracking_rmse'):
        vals=[r['goal_error'][key] for r in playback if key in r['goal_error']]
        if vals: metrics[key]=dict(mean=float(np.mean(vals)),max=float(np.max(vals)))
    return dict(metrics=metrics,reference_consumed=any(r['event']=='REFERENCE_CONSUMED' for r in rows),
        internal_stabilization=any(r['event']=='REFERENCE_CONSUMED' for r in rows),
        final_error=steps[-1]['goal_error'] if steps else {},
        final_predicates=steps[-1]['predicates'] if steps else {},
        invariant_violations=sorted(set(v for r in steps for v in r.get('invariant_violations',[]))))


def _run_instruction(runtime,adapter,instruction,trial_dir,stage,negative_control=False):
    source_terminal=runtime.last_outcome.terminal if runtime.last_outcome else None
    before=adapter.snapshot()
    history_before=policy_history_fingerprint(adapter)
    live_before=live_state_fingerprint(adapter)
    audit=dict(stage=stage,source_behavior=runtime.last_outcome.instruction.opcode if runtime.last_outcome else None,
        source_execution_id=runtime.last_outcome.evidence_path if runtime.last_outcome else None,
        source_terminal_snapshot=asdict(source_terminal) if source_terminal else None,
        target_behavior=instruction.opcode,target_asset=instruction.target_id,
        target_entry_snapshot=asdict(before),entry_history=history_before,entry_live=live_before,
        reference_alignment='FIXED_WORLD_V1',
        reference_mismatch=reference_entry_mismatch(adapter,before) if instruction.opcode=='TRACK_MOTION' else None)
    if source_terminal:
        audit.update(verify_handoff(source_terminal,before,history_before,history_before))
        if not audit['source_target_snapshot_equal_excluding_wall_time']:
            raise RuntimeError('SOURCE_TARGET_STATE_DISCONTINUITY')
    try:
        path=runtime.start_instruction(instruction,evaluation_mode=True)
    except RuntimeError as error:
        if not str(error).startswith('PRECONDITION_REJECTED'):
            raise
        audit.update(status='PRECONDITION_REJECTED',reason=str(error),entry_predicates='runtime physical checks',
                     exit_live=live_state_fingerprint(adapter),exit_history=policy_history_fingerprint(adapter))
        if live_state_fingerprint(adapter)!=live_before or policy_history_fingerprint(adapter)!=history_before:
            raise RuntimeError('REJECTED_START_MUTATED_LIVE_STATE_OR_HISTORY')
        return audit
    audit['execution_id']=path
    after_start=adapter.snapshot()
    audit['entry_predicates']='accepted by runtime'
    audit['start_live_unchanged']=live_state_fingerprint(adapter)==live_before
    audit['start_history_unchanged']=policy_history_fingerprint(adapter)==history_before
    if not audit['start_live_unchanged'] or not audit['start_history_unchanged']:
        raise RuntimeError('START_MUTATED_LIVE_STATE_OR_HISTORY')
    if negative_control:
        adapter.current_z=adapter.poses.get('default_pose')['latent']
        original_step=adapter.step
        def step_with_stand(elapsed,tracking=False):
            return original_step(elapsed,tracking=False)
        adapter.step=step_with_stand
    try:
        result=None
        while result is None: result=runtime.tick()
    finally:
        if negative_control: adapter.step=original_step
    audit.update(status=result.status,reason=result.termination_reason,
        exit_snapshot=asdict(result.terminal),exit_live=live_state_fingerprint(adapter),
        exit_history=policy_history_fingerprint(adapter),sim_elapsed_s=result.elapsed_sim_s,
        execution_completion=result.status=='SUCCESS',
        quality=stage_quality(result.evidence_path))
    if negative_control: audit['latent_override']='STAND_LATENT_BENCHMARK_ONLY'
    return audit


def run_trial(adapter,contract_config,phase3_config,trial,output):
    trial_dir=Path(output)/'trials'/trial['trial_id']
    trial_dir.mkdir(parents=True,exist_ok=True)
    guard=LiveMutationGuard(adapter)
    started_wall=time.monotonic()
    report=dict(trial=trial,stages=[],status=None,initialization_audit=None)
    try:
        initial,reasons=initialize_episode(adapter,trial,phase3_config,guard)
        report['initial_snapshot']=asdict(initial)
        report['initialization_audit']=dict(guard.calls,initial_history=policy_history_fingerprint(adapter),
            initial_live=live_state_fingerprint(adapter),reasons=reasons)
        if reasons:
            report.update(status='INVALID_INITIALIZATION',failure_reason=','.join(reasons))
            return report
        negative=trial['behavior']=='NEGATIVE_CONTROL'
        from bodyisa.core import issue_local_evaluation_permit
        runtime=ExecutionRuntime(adapter,contract_config,trial_dir/'executions',register_validation=not negative,
            evaluation_permit=issue_local_evaluation_permit(trial['trial_id']))
        # A fresh Runtime is an episode. Only first independent behavior can bootstrap.
        if trial['behavior']=='STAND': tasks=[('STAND',Instruction('STAND'))]
        elif trial['behavior'] in ('TRACK_MOTION','NEGATIVE_CONTROL'):
            tasks=[('TRACK_PLAYBACK',Instruction('TRACK_MOTION',adapter.reference.reference_id))]
        else:
            tasks=[('STAND_INITIAL',Instruction('STAND'))]
            for n in range(trial['sequence_length']):
                tasks += [(f'TRACK_{n+1}',Instruction('TRACK_MOTION',adapter.reference.reference_id)),
                          (f'STAND_EXPLICIT_{n+1}',Instruction('STAND'))]
        for stage,instruction in tasks:
            # No live control or sim-time advance between completed execution and handoff.
            before_time=adapter.snapshot().sim_time
            if report['stages']:
                previous=report['stages'][-1]
                if (policy_history_fingerprint(adapter)!=previous['exit_history'] or
                    live_state_fingerprint(adapter)!=previous['exit_live']):
                    raise RuntimeError('INTER_STAGE_STATE_OR_HISTORY_DISCONTINUITY')
            stage_result=_run_instruction(runtime,adapter,instruction,trial_dir,stage,
                negative_control=negative and stage=='TRACK_PLAYBACK')
            report['stages'].append(stage_result)
            if stage_result['status']=='PRECONDITION_REJECTED':
                report.update(status='PRECONDITION_REJECTED',failure_stage=stage,
                              failure_reason=stage_result['reason'])
                break
            if stage_result['status']!='SUCCESS':
                report.update(status='EXECUTION_FAILED',failure_stage=stage,
                              failure_reason=stage_result['reason'])
                break
            if adapter.snapshot().sim_time < before_time:
                raise RuntimeError('SIM_TIME_WENT_BACKWARD')
        else:
            report['status']='SUCCESS'
        return report
    except Exception:
        report.update(status='HARNESS_ERROR',failure_reason=traceback.format_exc())
        if 'runtime' in locals() and runtime.active:
            try: runtime.cancel()
            except Exception: pass
        return report
    finally:
        report['wall_s']=time.monotonic()-started_wall
        report['final_guard_calls']=dict(guard.calls)
        report['final_live']=live_state_fingerprint(adapter)
        report['final_history']=policy_history_fingerprint(adapter)
        guard.unlock_and_restore()
        (trial_dir/'trial.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


def wilson(success,n,z=1.959963984540054):
    if n==0:return None
    p=success/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half,center+half]


def aggregate(config,manifest,results):
    groups={}
    for trial in manifest:
        group=trial['group']
        if group not in groups:
            groups[group]=dict(requested=0,not_run=0,invalid_initialization=0,
                precondition_rejected=0,started=0,contract_success=0,timeout=0,
                invariant_violation=0,policy_runtime_error=0,harness_error=0,
                execution_failed=0,first_failure_stage={},failure_breakdown={},
                sampled_scenarios={})
        g=groups[group]; g['requested']+=1
        g['sampled_scenarios'][trial['scenario_class']]=g['sampled_scenarios'].get(trial['scenario_class'],0)+1
        result=results.get(trial['trial_id'])
        if not result: g['not_run']+=1;continue
        status=result['status']
        if status=='INVALID_INITIALIZATION':g['invalid_initialization']+=1
        elif status=='PRECONDITION_REJECTED':g['precondition_rejected']+=1
        elif status=='HARNESS_ERROR':g['harness_error']+=1
        else:
            g['started']+=1
            if status=='SUCCESS':g['contract_success']+=1
            else:g['execution_failed']+=1
        if status not in ('SUCCESS','INVALID_INITIALIZATION'):
            stage=result.get('failure_stage','INITIALIZATION_OR_HARNESS')
            g['first_failure_stage'][stage]=g['first_failure_stage'].get(stage,0)+1
            reason=result.get('failure_reason',status)
            if 'TIMEOUT' in reason:g['timeout']+=1
            elif 'INVARIANT_VIOLATION' in reason:g['invariant_violation']+=1
            elif status=='EXECUTION_FAILED':g['policy_runtime_error']+=1
            g['failure_breakdown'][reason]=g['failure_breakdown'].get(reason,0)+1
    for g in groups.values():
        g['success_per_started']=g['contract_success']/g['started'] if g['started'] else None
        g['wilson95_for_preregistered_mixture']=wilson(g['contract_success'],g['started'])
        g['end_to_end_success_per_requested']=g['contract_success']/g['requested']
    return groups


def build_envelopes(config,contract_hash,identity,manifest,results,aggregates):
    envelopes=[]
    for group,g in aggregates.items():
        if g['contract_success']<1 or group=='negative_control':continue
        rows=[(t,results.get(t['trial_id'])) for t in manifest if t['group']==group]
        support=tuple(dict(trial_id=t['trial_id'],scenario_class=t['scenario_class'],
            proposal=t['proposal'],entry_snapshot=r.get('initial_snapshot'),outcome=r['status'])
            for t,r in rows if r and r['status']=='SUCCESS')
        behavior='TRACK_MOTION' if group.startswith('track_') else 'STAND' if group.startswith('stand_') else 'CHAIN'
        if behavior=='CHAIN':continue  # sequence evidence reported separately, not a new callable behavior
        paths=tuple(str(Path('trials')/t['trial_id']/'trial.json') for t,r in rows if r)
        errors=[s.get('quality',{}).get('final_error',{}) for t,r in rows if r for s in r.get('stages',[]) if s.get('status')=='SUCCESS']
        times=[s.get('sim_elapsed_s') for t,r in rows if r for s in r.get('stages',[]) if s.get('status')=='SUCCESS']
        qualities=[s.get('quality',{}).get('metrics',{}) for t,r in rows if r for s in r.get('stages',[]) if s.get('status')=='SUCCESS']
        envelopes.append(ValidationEnvelope(behavior,None if behavior=='STAND' else config['asset']['target_id'],
            None if behavior=='STAND' else config['asset']['sha256'],identity['checkpoint_sha256'],
            identity['mujoco'],identity['adapter_version'],contract_hash,'SIM_ONLY',
            dict(scenario_classes=config['scenario_distribution']['classes'],
                 note='proposal only; no continuous region inferred'),support,
            config['entry_conditions'], 'reference_first' if behavior=='TRACK_MOTION' else 'reset_default',
            None,'RESET_ACTOR_HISTORY','FIXED_WORLD_V1',
            'heldout_validation' if 'heldout' in group else 'exploration',
            {k:v for k,v in g.items() if k in ('requested','not_run','invalid_initialization',
              'precondition_rejected','started','contract_success','timeout','invariant_violation',
              'policy_runtime_error','harness_error','execution_failed')},
            g['failure_breakdown'],dict(samples=errors),dict(samples=times),
            dict(samples=qualities),paths))
    return envelopes
