import copy
import json
from pathlib import Path
from collections import Counter
from .phase3 import make_manifest

def manifest(phase3_config):
    cfg=copy.deepcopy(phase3_config)
    cfg['seeds']=dict(exploration=61000,heldout_validation=62000,transition=63000,
                      composition=64000,negative_control=65000)
    cfg['trials']={k:0 for k in cfg['trials']}
    cfg['trials'].update(track_exploration=16,transition_s1=16,composition_s3=8)
    base=make_manifest(cfg)
    out=[]
    for trial in base:
        trial['split']='D2_INDEPENDENT_V2'
        trial['control']='NORMAL'
        if trial['group']=='track_exploration':trial['group']='reference_first'
        elif trial['group']=='transition_s1':trial['group']='transition_s1_v2'
        else:trial['group']='composition_s3_v2'
        out.append(trial)
    for kind in ('STAND','FROZEN_250','TIME_SHIFT_PLUS_100'):
        for i in range(8):
            trial=copy.deepcopy(out[i])
            trial['trial_id']=f'{kind.lower()}_control_{i:03d}'
            trial['group']=f'{kind.lower()}_control'
            trial['control']=kind
            trial['behavior']='NEGATIVE_CONTROL'
            trial['paired_reference_trial']=out[i]['trial_id']
            out.append(trial)
    assert len(out)==64 and len({t['trial_id'] for t in out})==64
    return out


def planned_stages(trial):
    kind=trial['behavior']
    if kind=='STAND':return ['STAND']
    if kind in ('TRACK_MOTION','NEGATIVE_CONTROL'):return ['TRACK_PLAYBACK']
    stages=['STAND_INITIAL']
    for n in range(1,trial['sequence_length']+1):
        stages.extend([f'TRACK_{n}',f'STAND_EXPLICIT_{n}'])
    return stages

def audit_episode(trial,row,execution_dir=None):
    plan=planned_stages(trial)
    actual=[s['stage'] for s in row['stages']]
    invalid=row['status']=='INVALID_INITIALIZATION'
    fail_stage=row.get('failure_stage')
    if invalid:expected=[]
    elif fail_stage in plan:expected=plan[:plan.index(fail_stage)+1]
    else:expected=plan
    expected_edges=list(zip(expected[:-1],expected[1:]))
    observed_edges=list(zip(actual[:-1],actual[1:]))
    expected_counts=Counter(expected_edges)
    observed_counts=Counter(observed_edges)
    missing=list((expected_counts-observed_counts).elements())
    duplicate=list((observed_counts-expected_counts).elements())
    unexpected=[edge for edge in observed_edges if edge not in expected_counts]
    matched=[]
    unverifiable=[]
    for source_name,target_name in expected_edges:
        matches=[i for i,edge in enumerate(observed_edges) if edge==(source_name,target_name)]
        if len(matches)!=1:continue
        i=matches[0]
        source,target=row['stages'][i:i+2]
        checks={
            'source_execution_id':target.get('source_execution_id')==source.get('execution_id'),
            'source_terminal_vs_target_entry':source.get('exit_snapshot') is not None and
                all(source['exit_snapshot'].get(k)==target.get('target_entry_snapshot',{}).get(k)
                    for k in source['exit_snapshot'] if k!='wall_time'),
            'sim_time':source.get('exit_snapshot',{}).get('sim_time')==target.get('target_entry_snapshot',{}).get('sim_time'),
            'actor_and_reference_history':source.get('exit_history',{}).get('sha256')==target.get('entry_history',{}).get('sha256'),
            'last_action':source.get('exit_history',{}).get('last_action')==target.get('entry_history',{}).get('last_action'),
            'live_integration_state':source.get('exit_live')==target.get('entry_live'),
            'no_live_reset_or_set_state_after_init':row['final_guard_calls']['blocked_reset']==0 and
                row['final_guard_calls']['blocked_set_state']==0
        }
        if execution_dir is not None:
            path=Path(target.get('execution_id',''))
            if not path.is_file():
                checks['target_execution_log_exists']=False
                unverifiable.append(dict(edge=[source_name,target_name],missing='target execution JSONL'))
            else:
                events=[json.loads(line) for line in path.read_text().splitlines()]
                initial=next((e for e in events if e.get('event')=='INITIAL'),None)
                checks['target_initial_log_snapshot']=(initial is not None and
                    all(initial['snapshot'].get(k)==target.get('target_entry_snapshot',{}).get(k)
                        for k in initial['snapshot'] if k!='wall_time'))
        matched.append(dict(source=source_name,target=target_name,checks=checks,
                            all_checks_passed=all(checks.values())))
        if not all(checks.values()):
            unverifiable.append(dict(edge=[source_name,target_name],failed=[k for k,v in checks.items() if not v]))
    internal=[]
    for stage in row['stages']:
        if stage.get('target_behavior')!='TRACK_MOTION':continue
        evidence=stage.get('execution_id')
        if evidence and Path(evidence).is_file():
            events=[json.loads(line) for line in Path(evidence).read_text().splitlines()]
            transitions=[e for e in events if e.get('event')=='REFERENCE_CONSUMED']
            internal.append(dict(stage=stage['stage'],playback_to_terminal_stand_events=len(transitions),
                                 reference_consumed=bool(stage.get('quality',{}).get('reference_consumed'))))
        else:
            internal.append(dict(stage=stage['stage'],playback_to_terminal_stand_events=None,
                                 status='UNVERIFIABLE_EXECUTION_LOG'))
    return dict(trial_id=trial['trial_id'],status=row['status'],planned_instruction_sequence=plan,
        actual_instruction_sequence=actual,planned_external_handoffs=len(plan)-1,
        expected_handoffs_for_actual_execution=len(expected_edges),
        observed_external_handoffs=len(observed_edges),matched_handoffs=len(matched),
        missing_handoffs=[list(x) for x in missing],duplicate_handoffs=[list(x) for x in duplicate],
        unexpected_handoffs=[list(x) for x in unexpected],handoff_checks=matched,
        internal_playback_to_stand=internal,
        explicit_stand_instructions=sum(x.startswith('STAND_EXPLICIT') for x in actual),
        episode_initialization=row.get('initialization_audit'),
        skipped_after_failure=plan[len(expected):] if row['status']!='SUCCESS' else [],
        exclusion_reason=row.get('failure_reason') if invalid else None,
        unverifiable=unverifiable)

