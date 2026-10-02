"""Explicit offline migration review. Never imported by session_factory or Runtime."""
import argparse
import ast
from pathlib import Path
import json
import shutil
from bodyisa.runtime.evidence import digest, file_hash, inside

ROOT = Path(__file__).resolve().parents[1]
OLD_ANCHOR = 'analysis/phase32_trust/p32_a_v1b_20260930_final.json'
OLD_DIRECTORY = 'analysis/phase32_evidence/p32_a_v1b_20260930/final_reviewed'


def baseline_inventory(root):
    expected = json.loads((ROOT/'evaluation/baseline_inventory.json').read_text())
    for name, record in expected.items():
        path = inside(root, root/name)
        if not path.is_file() or file_hash(path) != record['sha256']:
            raise RuntimeError('BASELINE_CHANGED_OR_MISSING: '+name)
    return expected


def symbol(path, qualified):
    node = ast.parse(path.read_text(encoding='utf-8'))
    for part in qualified.split('.'):
        node = next(x for x in node.body if getattr(x, 'name', None) == part)
    return node


def compatibility(baseline):
    inventory = baseline_inventory(baseline)
    mapping = []
    exact = [
        ('bodyisa/core.py','ExecutionContract','bodyisa/core/contracts.py','ExecutionContract'),
        ('bodyisa/core.py','TransitionContract','bodyisa/core/contracts.py','TransitionContract'),
        ('bodyisa/core.py','ExecutionRuntime.tick','bodyisa/runtime/session.py','ExecutionRuntime.tick'),
        ('bodyisa/core.py','ExecutionRuntime._finish','bodyisa/runtime/session.py','ExecutionRuntime._finish'),
        ('bodyisa/bfm_adapter.py','ReferenceMotion','bodyisa/adapters/g1/adapter.py','ReferenceMotion'),
        *[('bodyisa/bfm_adapter.py','BFMAdapter.'+name,'bodyisa/adapters/g1/adapter.py','G1Adapter.'+name)
          for name in ('snapshot','step','prepare','use_stand','tracking_metrics')],
        ('bodyisa/admission.py','EntryPolicy','bodyisa/runtime/admission.py','EntryPolicy'),
        ('bodyisa/admission.py','SourceReceipt','bodyisa/runtime/admission.py','SourceReceipt'),
        ('bodyisa/track_contract_v2.py','JointTrajectoryObserver','bodyisa/profiles/g1/track_contract_v2.py','JointTrajectoryObserver')]
    for old, old_symbol, new, new_symbol in exact:
        equal = ast.dump(symbol(baseline/old, old_symbol), include_attributes=False) == ast.dump(symbol(ROOT/new, new_symbol), include_attributes=False)
        mapping.append(dict(old_file=old,old_symbol=old_symbol,new_file=new,new_symbol=new_symbol,
            change='mechanical extraction / class relocation; method AST identical',ast_equal=equal,
            old_sha256=file_hash(baseline/old),new_sha256=file_hash(ROOT/new)))
    configs = [('bodyisa/config.json','bodyisa/profiles/g1/physical.json'),
               ('analysis/phase31_v2_contract.json','bodyisa/profiles/g1/track_v2.json'),
               ('analysis/phase32_entry_policy.json','bodyisa/profiles/g1/entry_policy.json')]
    unchanged = {old:file_hash(baseline/old)==file_hash(ROOT/new) for old,new in configs}
    changes = [
        dict(old_file='bodyisa/core.py',new_file='bodyisa/runtime/session.py',
             responsibility='initialize_episode delegates same reset/set_state to G1Adapter; actual_identity delegates to adapter; target catalog and observer supplied by adapter',
             verification='37 migrated legacy assertions + injection/protocol regression + required new ordinary JSONL smoke'),
        dict(old_file='bodyisa/evidence.py',new_file='bodyisa/adapters/g1/identity.py',
             responsibility='actual loaded model/simulator/reference/latent attestation retained; paths are explicit config; source coverage expands to all package and external BFM Python sources',
             verification='all non-source historical backend identities must be exactly equal at local review'),
        dict(old_file='bodyisa/evidence.py',new_file='bodyisa/runtime/evidence.py',
             responsibility='data-only verifier now compares capability to attested adapter catalog; digest pinning, path bounds, payload hashes, review source and atomic import retained',
             verification='legacy tamper/scope/atomicity regressions, final migration bundle loading'),
        dict(old_file='analysis/phase3_evaluate.py',new_file='bodyisa/session_factory.py',
             responsibility='same frozen cache installation moved to explicit local factory; no runner import',
             verification='loaded cache tensor attestation and new JSONL smoke')]
    for change in changes:
        change.update(old_sha256=file_hash(baseline/change['old_file']),new_sha256=file_hash(ROOT/change['new_file']))
    return dict(schema='BODYISA_MIGRATION_AUDIT_V1',baseline_inventory_sha256=file_hash(ROOT/'evaluation/baseline_inventory.json'),
        baseline_files_verified=len(inventory),mapping=mapping,changed_responsibilities=changes,
        profile_bytes_unchanged=unchanged,status='PASS' if all(x['ast_equal'] for x in mapping) and all(unchanged.values()) else 'FAIL')


def verify_historical_bundle(baseline, label, new_identity):
    """Read historical data, not historical executable objects or claimed approved flags."""
    anchor_path=baseline/OLD_ANCHOR
    anchor=json.loads(anchor_path.read_text())
    path=baseline/OLD_DIRECTORY/(label+'_bundle.json')
    pinned=file_hash(path);record=anchor['approved_bundles'][pinned]
    if record['status']!='APPROVED' or inside(baseline/'analysis',baseline/'analysis'/record['path'])!=path.resolve():
        raise RuntimeError('HISTORICAL_TRUST_RECORD_MISMATCH')
    bundle=json.loads(path.read_text())
    for name,value in bundle['identity'].items():
        if name!='source_sha256' and new_identity[name]!=value:
            raise RuntimeError('PHYSICAL_OR_SEMANTIC_IDENTITY_CHANGED: '+name)
    for ref in bundle['payloads']:
        target=inside(baseline/'analysis',baseline/'analysis'/ref['path'])
        if file_hash(target)!=ref['sha256']:raise RuntimeError('HISTORICAL_PAYLOAD_HASH: '+ref['path'])
    review=bundle['review'];review_path=inside(baseline/'analysis',baseline/'analysis'/review['path'])
    if file_hash(review_path)!=review['sha256'] or review['sha256']!=record['review_sha256']:
        raise RuntimeError('HISTORICAL_REVIEW_DIGEST')
    audit=json.loads(review_path.read_text())
    if audit['status']!='APPROVED' or audit['reviewer']!='LOCAL_AUTOMATED_REVIEW' or not all(audit['criteria'].values()):
        raise RuntimeError('HISTORICAL_REVIEW_STATUS')
    return bundle,pinned


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--regression-log',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();baseline=args.baseline_root.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('REFUSING_EXISTING_DEPLOYMENT_DIRECTORY')
    audit=compatibility(baseline)
    if audit['status']!='PASS':raise RuntimeError('MIGRATION_COMPATIBILITY_FAILED')
    regression=args.regression_log.read_text(encoding='utf-8')
    if not regression.rstrip().endswith('OK') or 'FAILED (' in regression or 'skipped=' in regression:
        raise RuntimeError('FULL_REGRESSION_PASS_REQUIRED')
    from bodyisa.session_factory import load_config,load_profiles,build_adapter
    config=load_config(args.config);physical,semantic,entry=load_profiles()
    adapter=build_adapter(config)
    try:identity=adapter.evidence_identity(physical,semantic,entry)
    finally:adapter.close()
    old=[verify_historical_bundle(baseline,label,identity) for label in ('stand','track')]
    output.mkdir(parents=True)
    def save(name,value):
        path=output/name;path.write_text(json.dumps(value,indent=2),encoding='utf-8');return path
    save('compatibility.json',audit)
    shutil.copyfile(args.regression_log,output/'regression.txt')
    save('identity.json',identity)
    save('smoke_plan.json',dict(schema='BODYISA_MIGRATION_SMOKE_PLAN_V1',seed=91000,episodes=1,
        path='one persistent ordinary JSONL process',baseline_run='p32_final_smoke_20261001_01',
        comparison_absolute_tolerance=dict(track_metric_rad=1e-5,consecutive_excess_s=1e-9,
            elapsed_sim_s=.04,root_displacement_m=.001,heading_change_rad=.001),
        outcomes='three independent SUCCESS; 500 semantic samples/PASS; exit SUCCESS; CLOSED and repeated TRACK rejected',
        admission='no permit, no within-chain reset, same frozen thresholds'))
    review_id='BODYISA_032_MODULE_MIGRATION_REVIEW_V1'
    review=save('review.json',dict(status='APPROVED',review_id=review_id,reviewer='LOCAL_AUTOMATED_REVIEW',
        reason='Explicit finite-scope code migration from pinned Phase 3.2 final evidence; physical and semantic identities unchanged; required independent Bridge smoke is reported separately, never backfilled into old evidence.',
        identity_hash=digest(identity),entry_policy_hash=identity['entry_policy_hash'],
        criteria=dict(baseline_bytes_verified=True,method_ast_compatibility=True,profile_bytes_unchanged=True,
            full_regressions_pass=True,actual_non_source_backend_identity_matches=True),
        original_bundles={label:pair[1] for label,pair in zip(('stand','track'),old)},
        original_trust_anchor_sha256=file_hash(baseline/OLD_ANCHOR),compatibility_sha256=file_hash(output/'compatibility.json')))
    approved={}
    for label,(original,old_digest) in zip(('stand','track'),old):
        bundle=json.loads(json.dumps(original))
        refs=[]
        for ref in original['payloads']:
            src=baseline/'analysis'/ref['path'];relative=Path('historical')/ref['path'];dst=output/relative
            dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():shutil.copyfile(src,dst)
            refs.append(dict(ref,path=relative.as_posix()))
        for role,name in [('compatibility','compatibility.json'),('regression','regression.txt'),('migration_smoke_plan','smoke_plan.json')]:
            refs.append(dict(role=role,path=name,sha256=file_hash(output/name)))
        bundle.update(identity=identity,payloads=refs,review=dict(path='review.json',sha256=file_hash(review)),
            migration=dict(schema='BODYISA_MIGRATION_V1',original_bundle_sha256=old_digest,
                original_run='p32_a_v1b_20260930 + p32_b_v1_20260930; original versions only',
                compatibility_sha256=file_hash(output/'compatibility.json')))
        path=save(label+'_bundle.json',bundle)
        approved[file_hash(path)]=dict(status='APPROVED',path=path.name,review_id=review_id,review_sha256=file_hash(review))
    save('trust.json',dict(schema='BODYISA_LOCAL_TRUST_V1',approved_bundles=approved))
    print(json.dumps(dict(status='MIGRATION_REVIEW_APPROVED',output=str(output),identity_hash=digest(identity)),indent=2))


if __name__=='__main__':main()
