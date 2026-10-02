from dataclasses import dataclass, asdict
import json
import numpy as np

@dataclass(frozen=True)
class ValidationEnvelope:
    """Finite empirical evidence; never an interpolated operational guarantee."""
    behavior: str
    target_id: str | None
    asset_sha256: str | None
    checkpoint_sha256: str
    simulator_version: str
    adapter_version: str
    contract_hash: str
    scope: str
    candidate_region: dict
    empirical_support: tuple
    runtime_entry_conditions: dict
    initialization: str
    previous_behavior: str | None
    policy_history: str
    reference_alignment: str
    split: str
    counts: dict
    failure_breakdown: dict
    terminal_errors: dict
    completion_times_s: dict
    quality_metrics: dict
    evidence_paths: tuple
    semantic_profile: str = 'LEGACY_PROCESS'


class PoseRegistry:
    def __init__(self):
        self.entries = {}

    def register(self, pose_id, source, joints, latent_reference, latent):
        if latent is None or not np.isfinite(joints).all():
            raise ValueError('Pose must be encoded before registration')
        self.entries[pose_id] = dict(pose_id=pose_id, source=source,
            joint_target=np.asarray(joints).copy(), latent_reference=latent_reference,
            latent=latent, validation_status='NOT_PHYSICS_VALIDATED')

    def get(self, pose_id):
        return self.entries[pose_id]


class CapabilityRegistry:
    def __init__(self):
        self.entries = {}
        self.require_envelope = False
        self._verified_bundles={}

    def install_verified(self,verified):
        payload=verified.payload()
        name=payload['capability']['opcode']
        # Validation completed before this atomic assignment; legacy entries cannot grant this authority.
        self._verified_bundles[name]=verified

    def verified_authorization(self,name,target_id,profile,identity):
        verified=self._verified_bundles.get(name)
        if verified is None:return dict(authorized=False,reason='NO_TRUSTED_EVIDENCE')
        bundle=verified.payload();cap=bundle['capability']
        if (cap['target_id']!=target_id or cap['semantic_profile']!=profile or
            cap['scope']!='SIM_ONLY' or bundle['identity']!=identity):
            return dict(authorized=False,reason='EVIDENCE_SCOPE_MISMATCH')
        return dict(authorized=True,reason=None,bundle_sha256=verified.bundle_sha256,
                    capability=cap,entry_policy_hash=bundle['entry_policy_hash'],
                    provenance=json.loads(verified.provenance_json))

    def add(self, name, levels, implementation='IMPLEMENTED', **metadata):
        self.entries[name] = dict(name=name, evidence_levels=list(levels),
            implementation=implementation, validation_status='NOT_PHYSICS_VALIDATED',
            scope='SIM_ONLY', **metadata)

    def validate(self, name, outcome, config_hash, target_id):
        if outcome.status != 'SUCCESS':
            return
        entry = self.entries[name]
        if self.require_envelope:
            entry.setdefault('candidate_success_evidence',[]).append(dict(
                config_hash=config_hash,target_id=target_id,evidence_path=outcome.evidence_path,
                scope='SIM_ONLY',status='AWAITING_SCOPED_ENVELOPE'))
            return
        if 'PHYSICS_VALIDATED' not in entry['evidence_levels']:
            entry['evidence_levels'].append('PHYSICS_VALIDATED')
        entry.update(validation_status='PHYSICS_VALIDATED', config_hash=config_hash,
                     target_id=target_id, evidence_path=outcome.evidence_path,
                     initial=asdict(outcome.initial))

    def record_envelope(self, envelope):
        entry = self.entries[envelope.behavior]
        if envelope.scope != 'SIM_ONLY' or envelope.counts.get('contract_success',0)<1:
            return
        record=asdict(envelope)
        entry.setdefault('validation_envelopes',[]).append(record)
        if 'PHYSICS_VALIDATED' not in entry['evidence_levels']:
            entry['evidence_levels'].append('PHYSICS_VALIDATED')
        entry['validation_status']='PHYSICS_VALIDATED_WITH_SCOPED_EMPIRICAL_EVIDENCE'

    def withhold(self,name,reason):
        entry=self.entries[name]
        entry['exposure_status']='WITHHELD'
        entry['exposure_reason']=reason

    def authorize(self,name,config_hash,target_id,identity,asset_sha256=None,semantic_profile=None):
        entry=self.entries[name]
        if entry.get('implementation')!='IMPLEMENTED': raise RuntimeError('NOT_IMPLEMENTED')
        if entry.get('exposure_status')=='WITHHELD': raise RuntimeError('CAPABILITY_WITHHELD: '+entry['exposure_reason'])
        if name=='TRACK_MOTION' and semantic_profile!='JOINT_TRAJECTORY_V2':
            raise RuntimeError('LEGACY_TRACK_SEMANTICS_WITHHELD')
        if self.require_envelope:
            matches=[e for e in entry.get('validation_envelopes',[]) if
                e['contract_hash']==config_hash and e['target_id']==target_id and
                e['scope']=='SIM_ONLY' and e['checkpoint_sha256']==identity.get('checkpoint_sha256') and
                e['simulator_version']==identity.get('mujoco') and
                e['adapter_version']==identity.get('adapter_version') and
                e['asset_sha256']==asset_sha256 and
                (name!='TRACK_MOTION' or
                 (e.get('semantic_profile','LEGACY_PROCESS')==(semantic_profile or 'LEGACY_PROCESS') and
                  (semantic_profile!='JOINT_TRAJECTORY_V2' or
                   e.get('reference_alignment')==identity.get('reference_preprocessing'))))]
            if not matches: raise RuntimeError('NO_MATCHING_VALIDATION_ENVELOPE')
        elif 'PHYSICS_VALIDATED' not in entry['evidence_levels'] or entry.get('config_hash')!=config_hash or entry.get('target_id')!=target_id:
            raise RuntimeError('NOT_PHYSICS_VALIDATED for this contract/target')

    def evidence_scope(self,name,config_hash,target_id,identity,asset_sha256=None,semantic_profile=None):
        self.authorize(name,config_hash,target_id,identity,asset_sha256,semantic_profile)
        return self.entries[name].get('validation_envelopes',[])

    def describe_capabilities(self, debug=False):
        return [dict(v) for v in self.entries.values() if debug or
                (v['implementation'] == 'IMPLEMENTED' and
                 'PHYSICS_VALIDATED' in v['evidence_levels'] and
                 v.get('exposure_status')!='WITHHELD' and
                 (not self.require_envelope or bool(v.get('validation_envelopes'))) and
                 (v['name']!='TRACK_MOTION' or any(
                     e.get('semantic_profile')=='JOINT_TRAJECTORY_V2' and
                     e.get('counts',{}).get('contract_success',0)>0
                     for e in v.get('validation_envelopes',[]))))]


