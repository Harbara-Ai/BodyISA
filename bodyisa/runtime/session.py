"""Shared synchronous runtime. Concrete adapters are injected by session_factory."""
from dataclasses import asdict
from pathlib import Path
import hashlib,json,math,uuid
import numpy as np
from bodyisa.core import *
from bodyisa.adapters.base import BodyAdapter
from .admission import EntryPolicy,SourceReceipt
from .evidence import canonical,digest,verify_bundle
from .permits import _LOCAL_EVALUATION_MARKER

class ExecutionRuntime:
    def __init__(self, adapter: BodyAdapter, config, evidence_dir, register_validation=True,
                 evaluation_permit=None, track_v2_config=None,entry_policy=None,
                 allowed_evidence_root=None,trusted_registry_path=None):
        self.adapter, self.config = adapter, json.loads(canonical(config))
        self.config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.active = None
        self.last_outcome = None
        self.mode = 'UNKNOWN'
        self.counter = 0
        self.register_validation=register_validation
        self.evaluation_permit=evaluation_permit
        self.track_v2_config=json.loads(canonical(track_v2_config)) if track_v2_config else None
        self.semantic_hash=(hashlib.sha256(json.dumps(track_v2_config,sort_keys=True).encode()).hexdigest()
                            if track_v2_config else None)
        self.entry_policy=EntryPolicy(entry_policy) if entry_policy else None
        self.allowed_evidence_root=allowed_evidence_root
        self.trusted_registry_path=trusted_registry_path
        self._backend_identity=None
        self.session_id=str(uuid.uuid4())
        self.episode_id=None
        self.evidence_revision=0
        self._source_receipt=None
        self._initial_context=None
        self._entry_phase='UNINITIALIZED'

    def _audit_admission(self,event,record):
        def clean(value):
            if isinstance(value,float) and not math.isfinite(value):return str(value)
            if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [clean(v) for v in value]
            return value
        with (self.evidence_dir/'admission.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(clean(dict(event=event,session_id=self.session_id,
                episode_id=self.episode_id,evidence_revision=self.evidence_revision,**record)),allow_nan=False)+'\n')

    def initialize_episode(self,episode_id,joint_delta=None):
        """Trusted local setup entry, absent from Instruction; one initialization per Runtime."""
        if self.active is not None:raise RuntimeError('BUSY')
        if not self.entry_policy or self._entry_phase!='UNINITIALIZED':
            raise RuntimeError('EPISODE_INITIALIZATION_NOT_ALLOWED')
        delta=self.adapter.initialize_nominal(joint_delta, self.entry_policy.config['initial_joint_delta_max_rad'])
        self.episode_id=episode_id
        self._entry_phase='INITIALIZED'
        self._initial_context=self.adapter.context_identity()
        self._audit_admission('EPISODE_INITIALIZED',dict(snapshot=asdict(self.get_state()),
            context=self._initial_context,joint_delta_rad=delta.tolist(),entry_policy_hash=self.entry_policy.identity))
        return self.get_state()

    def load_evidence(self,bundle_path):
        if self.active is not None:raise RuntimeError('BUSY: evidence import requires idle Runtime')
        try:
            if not self.entry_policy or not self.allowed_evidence_root or not self.trusted_registry_path:
                raise RuntimeError('LOCAL_TRUST_CONFIGURATION_REQUIRED')
            identity=self.adapter.evidence_identity(self.config,self.track_v2_config,self.entry_policy.config)
            verified=verify_bundle(bundle_path,self.allowed_evidence_root,self.trusted_registry_path,identity)
        except Exception as error:
            self._audit_admission('EVIDENCE_IMPORT_REJECTED',dict(bundle_path=str(bundle_path),reason=str(error)))
            raise
        self.adapter.capabilities.install_verified(verified)
        self._backend_identity=json.loads(canonical(identity))
        self.evidence_revision+=1
        self._source_receipt=None
        self._audit_admission('EVIDENCE_IMPORTED',dict(bundle_sha256=verified.bundle_sha256,
            provenance=json.loads(verified.provenance_json),identity_hash=digest(identity)))
        return dict(bundle_sha256=verified.bundle_sha256,capability=verified.payload()['capability'])

    def _static_authorization(self,instruction,evaluation_mode):
        if evaluation_mode:
            if self.evaluation_permit is None or getattr(self.evaluation_permit,'_marker',None) is not _LOCAL_EVALUATION_MARKER:
                return dict(authorized=False,reason='EVALUATION_PERMIT_REQUIRED')
            return dict(authorized=True,reason=None,candidate=True)
        if not self.entry_policy or self._backend_identity is None:
            return dict(authorized=False,reason='NO_TRUSTED_EVIDENCE')
        identity=dict(self._backend_identity)
        identity.update(physical_contract_hash=digest(self.config),semantic_contract_hash=digest(self.track_v2_config),
                        entry_policy_hash=self.entry_policy.identity)
        profile='STAND_REGION_V1' if instruction.opcode=='STAND' else instruction.parameters.get('contract_profile')
        return self.adapter.capabilities.verified_authorization(instruction.opcode,instruction.target_id,profile,identity)

    def preflight(self,instruction,evaluation_mode=False):
        decision=dict(request=asdict(instruction),eligible=False,reason=None,
                      entry_phase=self._entry_phase,entry_policy_hash=self.entry_policy.identity if self.entry_policy else None)
        if self.active is not None:return dict(decision,reason='BUSY')
        if instruction.opcode not in ('STAND','REACH_POSE','TRACK_MOTION'):
            return dict(decision,reason='NOT_IMPLEMENTED')
        if self.entry_policy:
            valid=instruction in self.adapter.supported_instructions
            if not valid:return dict(decision,reason='REQUEST_PROFILE_TARGET_PARAMETERS_REJECTED')
        static=self._static_authorization(instruction,evaluation_mode)
        decision['authorization']=static
        if not static['authorized']:return dict(decision,reason=static['reason'])
        s=self.get_state();decision['snapshot']=asdict(s)
        p,errors,violations,_=ExecutionContract(self.config,self.adapter.mass).evaluate(s,0,self.config['timeout_s'])
        decision['physical_values']=errors;decision['physical_predicates']=p
        if violations:return dict(decision,reason='PRECONDITION_REJECTED',invariant_violations=violations)
        if self.entry_policy:
            context=self.adapter.context_identity();decision['context']=context
            if self._source_receipt is not None:
                source_contract=self.semantic_hash if self._source_receipt.opcode=='TRACK_MOTION' else self.config_hash
                if self._source_receipt.contract_hash!=source_contract or self._source_receipt.terminal_sim_time!=s.sim_time:
                    return dict(decision,reason='SOURCE_RECEIPT_CONTRACT_OR_TIME_MISMATCH')
            entry=self.entry_policy.check(instruction.opcode,s,self._entry_phase,self._source_receipt,
                context,self.session_id,self.episode_id,self.evidence_revision,self._initial_context)
            decision['entry']=entry
            if not entry['eligible']:return dict(decision,reason=entry['reason'])
            if instruction.opcode=='TRACK_MOTION' or self._entry_phase=='TRACK_DONE':
                if not all(p.values()):return dict(decision,reason='PRECONDITION_REJECTED: not standing')
        return dict(decision,eligible=True)

    def get_state(self):
        return self.adapter.snapshot()

    def get_robot_state(self):
        return RobotState(self.get_state(),self.mode)

    def describe_capabilities(self, debug=False):
        if debug:return self.adapter.capabilities.describe_capabilities(True)+[
            dict(v.payload()['capability'],static_authorization='APPROVED',bundle_sha256=v.bundle_sha256)
            for v in self.adapter.capabilities._verified_bundles.values()]
        result=[]
        for instruction in self.adapter.supported_instructions:
            static=self._static_authorization(instruction,False)
            if static['authorized']:
                current=self.preflight(instruction)
                result.append(dict(static['capability'],static_authorization='APPROVED',
                    evidence_version=static['bundle_sha256'],entry_policy_version=self.entry_policy.config['version'],
                    entry_eligible=current['eligible'],entry_reason=current['reason'],
                    required_predecessor='same Runtime successful external STAND' if instruction.opcode=='TRACK_MOTION' else 'controlled initialization or successful TRACK exit',
                    semantic_scope='joint trajectory and defined invariants and terminal stabilization' if instruction.opcode=='TRACK_MOTION' else 'standing region',
                    world_path_certified=False))
        return result

    def get_execution(self):
        return dict(active=asdict(self.active['instruction']) if self.active else None,
                    last_outcome=asdict(self.last_outcome) if self.last_outcome else None)

    def _log(self, record):
        self.active['file'].write(json.dumps(record, allow_nan=False)+'\n')
        self.active['file'].flush()

    def start_instruction(self, instruction, evaluation_mode=False):
        if self.entry_policy or not evaluation_mode:
            decision=self.preflight(instruction,evaluation_mode)
            if not decision['eligible']:
                self._audit_admission('START_REJECTED',decision)
                if decision['reason']=='NOT_IMPLEMENTED':raise NotImplementedError(decision['reason'])
                raise RuntimeError(decision['reason'])
            if self.entry_policy and self.adapter.context_identity()!=decision['context']:
                changed=dict(decision,eligible=False,reason='STATE_CHANGED_DURING_ADMISSION',current_context=self.adapter.context_identity())
                self._audit_admission('START_REJECTED',changed)
                raise RuntimeError(changed['reason'])
            self._audit_admission('START_ACCEPTED',decision)
        if self.active is not None: raise RuntimeError('BUSY: one active execution')
        if evaluation_mode and (self.evaluation_permit is None or
            getattr(self.evaluation_permit,'_marker',None) is not _LOCAL_EVALUATION_MARKER):
            raise RuntimeError('EVALUATION_PERMIT_REQUIRED')
        op = instruction.opcode
        if op not in ('STAND', 'REACH_POSE', 'TRACK_MOTION'):
            raise NotImplementedError(f'{op}: NOT_IMPLEMENTED / NOT_PHYSICS_VALIDATED')
        s = self.get_state()
        _, _, initial_violations, _ = ExecutionContract(self.config,self.adapter.mass).evaluate(s,0,self.config['timeout_s'])
        if initial_violations:
            raise RuntimeError('PRECONDITION_REJECTED: '+','.join(initial_violations))
        profile=instruction.parameters.get('contract_profile') if op=='TRACK_MOTION' else None
        if op=='TRACK_MOTION' and not evaluation_mode and profile!='JOINT_TRAJECTORY_V2':
            raise RuntimeError('LEGACY_TRACK_SEMANTICS_WITHHELD')
        if op=='TRACK_MOTION' and profile=='JOINT_TRAJECTORY_V2' and not self.track_v2_config:
            raise RuntimeError('TRACK_V2_CONFIG_REQUIRED')
        if not evaluation_mode and not self.entry_policy:
            asset_sha=(self.adapter.reference.source_hash if op=='TRACK_MOTION' else None)
            self.adapter.capabilities.authorize(op,
                self.semantic_hash if op=='TRACK_MOTION' else self.config_hash,
                instruction.target_id,self.adapter.identity,asset_sha,profile)
        transition = None
        if self.mode != op:
            if self.mode == 'UNKNOWN':
                if self.last_outcome is not None and op != 'STAND':
                    raise RuntimeError('PRECONDITION_REJECTED: STAND required after failure/cancel')
                if not evaluation_mode and op != 'STAND': raise RuntimeError('STAND required')
                transition = {'event': 'EVALUATION_BOOTSTRAP' if evaluation_mode else 'STAND_BOOTSTRAP'}
            else:
                tc = TransitionContract.between(self.mode,op,self.config['timeout_s'])
                if self.mode == 'STAND':
                    p, _, violations, _ = ExecutionContract(self.config,self.adapter.mass).evaluate(s,0,6)
                    if violations or not all(p.values()):
                        raise RuntimeError('PRECONDITION_REJECTED: transition entry not standing')
                transition = {'event': 'TRANSITION_ENTRY', **asdict(tc)}
        observer=None
        if profile=='JOINT_TRAJECTORY_V2':
            observer=self.adapter.create_observer(self.track_v2_config)
            observer.observe(s,0.0)
        objective, target, source = self.adapter.prepare(instruction)
        self.counter += 1
        path = self.evidence_dir / f'{self.counter:03d}_{op}.jsonl'
        self.active = dict(instruction=instruction, initial=s, start=s.sim_time,
            contract=ExecutionContract(self.config,self.adapter.mass,target if op=='REACH_POSE' else None),
            file=path.open('w',encoding='utf-8'), path=path, terminal_start=None,
            semantic_observer=None,semantic_failure=None,exit_outcome=None)
        if profile=='JOINT_TRAJECTORY_V2':
            self.active['semantic_observer']=observer
        self._log(dict(event='INITIAL', instruction=asdict(instruction), objective=asdict(objective),
            target=target.tolist() if target is not None else None, latent_source=source,
            snapshot=asdict(s), config=self.config, config_hash=self.config_hash,
            evaluation_mode=evaluation_mode,
            evaluation_run_id=self.evaluation_permit.run_id if evaluation_mode else None,
            semantic_profile=profile or 'LEGACY_PROCESS',semantic_config_hash=self.semantic_hash if profile else None,
            scope='SIM_ONLY', backend=self.adapter.identity))
        if transition: self._log(transition)
        return str(path)

    def _finish(self, reason, s):
        a = self.active
        status = 'SUCCESS' if reason=='SUCCESS' else 'CANCELLED' if reason=='CANCELLED' else 'FAILED'
        assessment=(a['semantic_observer'].assessment() if a['semantic_observer'] else None)
        outcome = ExecutionOutcome(a['instruction'],status,reason,s.sim_time-a['start'],
                                   a['initial'],s,str(a['path'].resolve()),assessment,
                                   a['exit_outcome'],a['semantic_failure']['primary_failure_class'] if a['semantic_failure'] else None)
        self._log(dict(event='TERMINAL', **asdict(outcome)))
        a['file'].close()
        self.mode = a['instruction'].opcode if status=='SUCCESS' else 'UNKNOWN'
        # TRACK completion has physically settled under STAND, so its destination is STAND.
        if status=='SUCCESS' and self.mode=='TRACK_MOTION': self.mode='STAND'
        if status=='SUCCESS' and self.register_validation:
            self.adapter.capabilities.validate(a['instruction'].opcode,outcome,
                self.semantic_hash if a['semantic_observer'] else self.config_hash,a['instruction'].target_id)
            if a['instruction'].opcode=='REACH_POSE':
                self.adapter.poses.get(a['instruction'].target_id)['validation_status']='PHYSICS_VALIDATED'
        self.active = None
        self.last_outcome = outcome
        if self.entry_policy:
            if status=='SUCCESS':
                if a['instruction'].opcode=='STAND':
                    self._entry_phase='STAND_DONE' if self._entry_phase=='INITIALIZED' else 'CLOSED'
                elif a['instruction'].opcode=='TRACK_MOTION':self._entry_phase='TRACK_DONE'
                self._source_receipt=SourceReceipt(self.session_id,self.episode_id,a['instruction'].opcode,
                    self.semantic_hash if a['semantic_observer'] else self.config_hash,status,str(a['path'].resolve()),
                    s.sim_time,canonical(self.adapter.context_identity()),self.evidence_revision)
            else:self._entry_phase='FAILED';self._source_receipt=None
            self._audit_admission('SOURCE_RECEIPT',dict(entry_phase=self._entry_phase,
                receipt=asdict(self._source_receipt) if self._source_receipt else None,outcome=asdict(outcome)))
        return outcome

    def cancel(self):
        if self.active is None: raise RuntimeError('No active execution')
        return self._finish('CANCELLED',self.get_state())

    def tick(self):
        if self.active is None: raise RuntimeError('No active execution')
        a = self.active
        op = a['instruction'].opcode
        elapsed = self.get_state().sim_time-a['start']
        if op=='TRACK_MOTION' and elapsed >= self.adapter.reference.duration-1e-9 and a['terminal_start'] is None:
            a['terminal_start'] = self.get_state().sim_time
            self.adapter.use_stand()
            a['contract'] = ExecutionContract(self.config,self.adapter.mass)
            self._log(dict(event='REFERENCE_CONSUMED',elapsed_sim_s=elapsed,
                transition=asdict(TransitionContract.between('TRACK_MOTION','STAND',self.config['terminal_timeout_s']))))
        terminal_control=a['terminal_start'] is not None
        try:
            self.adapter.step(elapsed, tracking=op=='TRACK_MOTION' and a['terminal_start'] is None)
        except Exception as error:
            self._log(dict(event='BACKEND_ERROR',error_type=type(error).__name__,message=str(error)))
            return self._finish('BACKEND_ERROR',self.get_state())
        s = self.get_state()
        elapsed = s.sim_time-a['start']
        terminal_elapsed = 0 if a['terminal_start'] is None else s.sim_time-a['terminal_start']
        p, errors, violations, reason = a['contract'].evaluate(s,
            terminal_elapsed if op=='TRACK_MOTION' else elapsed,
            self.config['terminal_timeout_s'] if op=='TRACK_MOTION' else self.config['timeout_s'])
        if op=='TRACK_MOTION' and a['terminal_start'] is None:
            reason = 'INVARIANT_VIOLATION' if violations else None
        if op=='TRACK_MOTION': errors.update(self.adapter.tracking_metrics(s,min(elapsed,self.adapter.reference.duration)))
        if a['semantic_observer'] is not None and a['terminal_start'] is None:
            assessment=a['semantic_observer'].observe(s,elapsed)
            if elapsed>=self.adapter.reference.duration-1e-9:
                assessment=a['semantic_observer'].finalize()
            errors['semantic_v2']=assessment['metrics']
            if assessment['status'] in ('FAIL','NOT_EVALUABLE') and not violations:
                a['semantic_failure']=assessment['failure'] or dict(primary_failure_class='OBSERVATION_UNAVAILABLE')
                a['terminal_start']=s.sim_time
                self.adapter.use_stand()
                a['contract']=ExecutionContract(self.config,self.adapter.mass)
                self._log(dict(event='SEMANTIC_FAILURE',assessment=assessment,
                    exit_policy='TRY_STAND_WITH_EXISTING_TERMINAL_TIMEOUT',sim_time=s.sim_time))
                reason=None
            elif assessment['status']=='PASS':
                self._log(dict(event='SEMANTIC_PASS',assessment=assessment,sim_time=s.sim_time))
        if a['semantic_observer'] is not None and a['terminal_start'] is not None and reason:
            a['exit_outcome']=reason
            if a['semantic_failure']:
                reason=a['semantic_failure']['primary_failure_class']
            elif reason=='SUCCESS' and a['semantic_observer'].assessment()['status']!='PASS':
                reason='TRACK_CONTRACT_VIOLATION'
        self._log(dict(event='STEP', elapsed_sim_s=elapsed, state=asdict(s),
            goal_error=errors, predicates=p, invariant_violations=violations,
            reference_consumed=terminal_control if op=='TRACK_MOTION' else None))
        return self._finish(reason,s) if reason else None


