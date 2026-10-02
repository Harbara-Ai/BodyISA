"""Data-only local evidence loading, with a separately configured trust anchor."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np

IDENTITY_KEYS={'checkpoint_sha256','actual_actor_normalizer_sha256','model_config_sha256',
    'model_init_kwargs_sha256','simulator_version','compiled_model_sha256','simulator_xml_sha256',
    'controller_parameters_sha256','control_dt_s','asset_sha256','reference_joints_sha256',
    'reference_preprocessing','track_latent_sha256','stand_latent_sha256','physical_contract_hash',
    'semantic_contract_hash','entry_policy_hash','source_sha256'}


def canonical(value):
    return json.dumps(value,sort_keys=True,allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def array_hash(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    value=np.ascontiguousarray(value)
    h=hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def inside(root,path):
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root):raise RuntimeError('EVIDENCE_PATH_OUTSIDE_ALLOWED_ROOT')
    return path


@dataclass(frozen=True)
class VerifiedBundle:
    payload_json: str
    bundle_sha256: str
    provenance_json: str

    def payload(self):return json.loads(self.payload_json)


def verify_bundle(bundle_path,allowed_root,anchor_path,identity):
    root=Path(allowed_root).resolve()
    path=inside(root,bundle_path)
    anchor=json.loads(Path(anchor_path).read_text(encoding='utf-8'))
    if anchor.get('schema')!='BODYISA_LOCAL_TRUST_V1':raise RuntimeError('TRUST_ANCHOR_SCHEMA')
    pinned=file_hash(path)
    record=anchor.get('approved_bundles',{}).get(pinned)
    if not record:raise RuntimeError('NO_TRUSTED_REVIEW_FOR_BUNDLE')
    if record.get('status')!='APPROVED':raise RuntimeError('REVIEW_WITHHELD_OR_REVOKED')
    if inside(root,root/record['path'])!=path:raise RuntimeError('TRUSTED_BUNDLE_PATH_MISMATCH')
    bundle=json.loads(path.read_text(encoding='utf-8'))
    required={'schema','capability','identity','entry_policy_hash','review','payloads','validation'}
    if not required.issubset(bundle) or bundle['schema']!='BODYISA_EVIDENCE_V1':
        raise RuntimeError('EVIDENCE_REQUIRED_FIELD_OR_SCHEMA')
    if not IDENTITY_KEYS.issubset(bundle['identity']) or not {'scope','requested','accepted','success','failures','rejections','source_run'}.issubset(bundle['validation']):
        raise RuntimeError('EVIDENCE_IDENTITY_OR_VALIDATION_FIELDS_MISSING')
    if bundle['validation']['scope']!='SIM_ONLY' or bundle['validation']['success']<1:
        raise RuntimeError('EVIDENCE_NO_SUCCESSFUL_SUPPORTED_PATH')
    cap=bundle['capability']
    valid=cap in identity.get('capabilities',[])
    if not valid:raise RuntimeError('CAPABILITY_PROFILE_TARGET_SCOPE_MISMATCH')
    if bundle['identity']!=identity or bundle['entry_policy_hash']!=identity['entry_policy_hash']:
        raise RuntimeError('CURRENT_BACKEND_CONTRACT_SCOPE_MISMATCH')
    roles=set();verified=[]
    for ref in bundle['payloads']:
        target=inside(root,root/ref['path'])
        if not target.is_file():raise RuntimeError('REQUIRED_EVIDENCE_MISSING:'+ref['path'])
        if file_hash(target)!=ref['sha256']:raise RuntimeError('PAYLOAD_HASH_MISMATCH:'+ref['path'])
        roles.add(ref['role']);verified.append(ref)
    if not {'stage_a_results','stage_a_manifest','frozen_config','entry_policy','compatibility','regression','execution_log'}.issubset(roles):
        raise RuntimeError('EVIDENCE_PAYLOAD_ROLES_MISSING')
    review=bundle['review']
    review_path=inside(root,root/review['path'])
    if file_hash(review_path)!=record['review_sha256'] or review['sha256']!=record['review_sha256']:
        raise RuntimeError('TRUSTED_REVIEW_DIGEST_MISMATCH')
    audit=json.loads(review_path.read_text(encoding='utf-8'))
    if audit.get('status')!='APPROVED' or audit.get('review_id')!=record['review_id']:
        raise RuntimeError('REVIEW_RECORD_MISMATCH')
    if audit.get('reviewer')!='LOCAL_AUTOMATED_REVIEW' or not audit.get('reason'):
        raise RuntimeError('REVIEW_SOURCE_UNKNOWN')
    if audit.get('identity_hash')!=digest(identity) or audit.get('entry_policy_hash')!=identity['entry_policy_hash']:
        raise RuntimeError('REVIEW_SCOPE_MISMATCH')
    if not audit.get('criteria') or not all(audit['criteria'].values()):raise RuntimeError('REVIEW_CRITERIA_FAILED')
    provenance=dict(path=str(path),review_id=record['review_id'],review_sha256=record['review_sha256'],
                    payloads=verified,trust_anchor_sha256=file_hash(anchor_path))
    return VerifiedBundle(canonical(bundle),pinned,canonical(provenance))
