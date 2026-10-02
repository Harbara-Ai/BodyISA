"""Read only one migration Bridge smoke; never runs actions or creates approval."""
import argparse
import json
import math
from pathlib import Path
from bodyisa.runtime.evidence import file_hash


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def analyze(run, baseline, deployment):
    setup=json.loads((run/'setup.json').read_text())
    plan=json.loads((deployment/'smoke_plan.json').read_text())
    tolerance=plan['comparison_absolute_tolerance']
    old=json.loads((baseline/'analysis/phase32_evidence/p32_final_smoke_20261001_01/example_result.json').read_text())
    events=read_jsonl(run/'admission.jsonl');wire=read_jsonl(run/'bridge.jsonl')
    responses=[x['value'] for x in wire if x['direction']=='response']
    checks={}
    def check(name,value): checks[name]=bool(value)
    physical=lambda s:{k:v for k,v in s.items() if k!='wall_time'}
    init=[e for e in events if e['event']=='EPISODE_INITIALIZED']
    imports=[e for e in events if e['event']=='EVIDENCE_IMPORTED']
    accepted=[e for e in events if e['event']=='START_ACCEPTED']
    receipts=[e for e in events if e['event']=='SOURCE_RECEIPT']
    rejected=[e for e in events if e['event']=='START_REJECTED']
    check('one_initialization',len(init)==1)
    check('one_session',len({r['session_id'] for r in responses})==1)
    check('ready_once',sum(r['status']=='READY' for r in responses)==1)
    check('no_permit',setup['evaluation_permit'] is None)
    check('same_fixed_seed_and_proposal',setup['seed']==plan['seed']==old['trial']['seed'] and setup['joint_delta_rad']==old['trial']['joint_delta_rad'])
    anchor=Path(setup['config']['trust_anchor']);trusted=json.loads(anchor.read_text())
    check('two_verified_migrated_imports',len(imports)==2 and all(e['provenance']['trust_anchor_sha256']==file_hash(anchor) and e['bundle_sha256'] in trusted['approved_bundles'] for e in imports))
    check('three_external_executions',len(accepted)==len(receipts)==3)
    check('no_harness_error',not any(r['status'] in ('ERROR','STARTUP_ERROR','FAILED') for r in responses))
    stages=[];handoffs=[];logs=[]
    for i,receipt in enumerate(receipts):
        result=receipt['outcome'];name=Path(result['evidence_path']).name;path=run/name
        log=read_jsonl(path);logs.append(log)
        initial=next(e for e in log if e['event']=='INITIAL');terminal=next(e for e in log if e['event']=='TERMINAL')
        steps=[e for e in log if e['event']=='STEP'];context=json.loads(receipt['receipt']['context_json'])
        check(name+'_real_success',result['status']=='SUCCESS' and terminal['status']=='SUCCESS')
        check(name+'_ordinary',initial['evaluation_mode'] is False and initial['evaluation_run_id'] is None)
        check(name+'_invariants',bool(steps) and all(not e['invariant_violations'] for e in steps))
        check(name+'_control_and_time_continuity',context['control_revision']-accepted[i]['context']['control_revision']==len(steps) and all(abs(e['state']['sim_time']-(result['initial']['sim_time']+(n+1)*.02))<1e-9 for n,e in enumerate(steps)))
        check(name+'_generation',context['reset_generation']==accepted[i]['context']['reset_generation']==init[0]['context']['reset_generation'])
        check(name+'_elapsed_vs_baseline',abs(result['elapsed_sim_s']-old['stages'][i]['outcome']['elapsed_sim_s'])<=tolerance['elapsed_sim_s'])
        stages.append(dict(execution_id=Path(name).stem,instruction=result['instruction'],status=result['status'],elapsed_sim_s=result['elapsed_sim_s'],steps=len(steps),evidence_file=name))
        if i:
            source=receipts[i-1];prev=json.loads(source['receipt']['context_json'])
            handoff=dict(source=stages[i-1]['execution_id'],target=stages[i]['execution_id'],
                sim_time=result['initial']['sim_time'],context_equal=prev==accepted[i]['context'],
                physical_equal=physical(source['outcome']['terminal'])==physical(initial['snapshot']))
            check('handoff_'+str(i),handoff['context_equal'] and handoff['physical_equal']);handoffs.append(handoff)
    track=receipts[1]['outcome'];semantic=track['semantic_assessment'];metrics=semantic['metrics'];oldmetrics=old['stages'][1]['outcome']['semantic_assessment']['metrics']
    check('semantic_complete_pass',semantic['status']=='PASS' and metrics['samples']==500 and not semantic['failure'] and not semantic['data_error'])
    check('track_exit_success',track['exit_stabilization_outcome']=='SUCCESS' and any(e['event']=='REFERENCE_CONSUMED' for e in logs[1]))
    for key in ('mean_of_frame_joint_rmse_rad','active_group_mean_of_frame_rmse_rad','max_25frame_mean_rad'):
        check('baseline_metric_'+key,abs(metrics[key]-oldmetrics[key])<=tolerance['track_metric_rad'])
    check('baseline_consecutive_duration',abs(metrics['max_consecutive_excess_s']-oldmetrics['max_consecutive_excess_s'])<=tolerance['consecutive_excess_s'])
    start=init[0]['snapshot'];end=receipts[-1]['outcome']['terminal']
    drift=math.dist(start['base_position_world'],end['base_position_world'])
    heading=abs(math.remainder(end['base_heading_world']-start['base_heading_world'],2*math.pi))
    olddrift=math.dist(old['initial_snapshot']['base_position_world'],old['final_snapshot']['base_position_world'])
    oldheading=abs(math.remainder(old['final_snapshot']['base_heading_world']-old['initial_snapshot']['base_heading_world'],2*math.pi))
    check('root_displacement_comparison',abs(drift-olddrift)<=tolerance['root_displacement_m'])
    check('heading_comparison',abs(heading-oldheading)<=tolerance['heading_change_rad'])
    check('closed',receipts[-1]['entry_phase']=='CLOSED' and all(not c['entry_eligible'] for c in responses[-1]['capabilities']))
    repeated=[r for r in responses if r.get('reason')=='UNSUPPORTED_TRACK_CONTEXT']
    check('closed_track_rejection_no_side_effect',len(repeated)==1 and repeated[0].get('unchanged') and repeated[0]['context_identity']==json.loads(receipts[-1]['receipt']['context_json']))
    check('only_expected_motion_rejection',len(rejected)==1 and rejected[0]['reason']=='UNSUPPORTED_TRACK_CONTEXT')
    query_states=[r for r in responses if r.get('tool') in ('get_state','describe_capabilities')]
    known=[init[0]['context']]+[json.loads(r['receipt']['context_json']) for r in receipts]
    check('queries_do_not_mutate',all(r['context_identity'] in known for r in query_states))
    check('eligible_after_real_stand',any(r.get('tool')=='describe_capabilities' and r['phase']=='STAND_DONE' and any(c['opcode']=='TRACK_MOTION' and c['entry_eligible'] for c in r['capabilities']) for r in responses))
    issues=[k for k,v in checks.items() if not v]
    return dict(status='SMOKE_PASS' if not issues else 'FAIL',run_id=run.name,seed=setup['seed'],
        stages=stages,metrics=metrics,track_exit=track['exit_stabilization_outcome'],handoffs=handoffs,
        root_displacement_m=drift,heading_change_abs_rad=heading,
        baseline_comparison=dict(root_displacement_m=olddrift,heading_change_abs_rad=oldheading,metrics=oldmetrics,tolerance=tolerance),
        closed=receipts[-1]['entry_phase'],checks=checks,issues=issues,
        artifact_sha256={p.name:file_hash(p) for p in (anchor,Path(setup['config']['stand_bundle']),Path(setup['config']['track_bundle']))},
        raw_log_sha256={p.name:file_hash(p) for p in sorted(run.glob('*.jsonl'))})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--deployment-root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:report=analyze(args.run_dir,args.baseline_root,args.deployment_root)
    except (KeyError,IndexError,StopIteration,FileNotFoundError,ValueError) as error:
        report=dict(status='UNVERIFIABLE',reason=type(error).__name__+': '+str(error))
    if args.output:
        if args.output.exists():raise FileExistsError('Refusing existing analysis output')
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='SMOKE_PASS' else 1


if __name__=='__main__':raise SystemExit(main())
