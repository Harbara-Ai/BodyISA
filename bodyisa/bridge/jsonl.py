"""One persistent ordinary Runtime; JSONL on stdin/stdout, diagnostics on stderr."""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
from bodyisa.core import Instruction

ARGUMENTS = {'describe_capabilities': set(), 'get_state': set(), 'STAND': set(),
             'TRACK_MOTION': {'target', 'contract_profile'}, 'get_execution': {'execution_id'}}


def parse_request(value):
    if not isinstance(value, dict) or set(value)-{'tool', 'arguments'} or 'tool' not in value:
        raise ValueError('INVALID_REQUEST_FIELDS')
    name = value['tool']
    if not isinstance(name, str) or name not in ARGUMENTS:
        raise ValueError('UNKNOWN_TOOL')
    args = value.get('arguments', {})
    if not isinstance(args, dict) or set(args) != ARGUMENTS[name]:
        raise ValueError('INVALID_ARGUMENTS')
    if name == 'TRACK_MOTION' and args != {'target': 'dance_prefix_500', 'contract_profile': 'JOINT_TRAJECTORY_V2'}:
        raise ValueError('UNSUPPORTED_TARGET_PROFILE')
    if name == 'get_execution' and (not isinstance(args['execution_id'], str) or not args['execution_id']):
        raise ValueError('INVALID_EXECUTION_ID')
    return name, args


def finite_data(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'value': None, 'available': False, 'reason': 'NONFINITE'}
    if isinstance(value, dict):
        return {k: finite_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_data(v) for v in value]
    return value


class Bridge:
    def __init__(self, runtime):
        self.runtime = runtime
        if runtime.evaluation_permit is not None:
            raise RuntimeError('ORDINARY_RUNTIME_REQUIRED')
        self.executions = {}
        self.faulted = False

    def context(self):
        r = self.runtime
        return dict(session_id=r.session_id, episode_id=r.episode_id,
            simulation_time_s=r.get_state().sim_time, phase=r._entry_phase,
            state=asdict(r.get_state()), context_identity=r.adapter.context_identity(),
            capabilities=r.describe_capabilities(), admission_log=str((r.evidence_dir/'admission.jsonl').resolve()))

    def ready(self):
        return dict(status='READY', protocol='BODYISA_JSONL_V1',
            embodiment='G1', controller='BFM-Zero', execution_backend='MuJoCo', scope='SIM_ONLY',
            joint_names=self.runtime.adapter.identity['joint_order'],
            units={'joint_position': 'rad', 'joint_velocity': 'rad/s', 'position': 'm', 'time': 's'},
            frames={'base': 'world', 'quaternion': 'wxyz'}, **self.context())

    def handle(self, request):
        try:
            name, args = parse_request(request)
        except ValueError as error:
            return dict(status='REJECTED', admission='NOT_ACCEPTED', reason=str(error), **self.context())
        r = self.runtime
        if name in ('get_state', 'describe_capabilities'):
            return dict(status='OK', tool=name, **self.context())
        if name == 'get_execution':
            outcome = self.executions.get(args['execution_id'])
            return dict(status='OK' if outcome else 'REJECTED', tool=name,
                reason=None if outcome else 'UNKNOWN_EXECUTION_ID', outcome=outcome, **self.context())
        if self.faulted:
            return dict(status='REJECTED', admission='NOT_ACCEPTED', reason='SESSION_FAULTED', **self.context())
        instruction = Instruction(name, args['target'], {'contract_profile': args['contract_profile']}) if name == 'TRACK_MOTION' else Instruction(name)
        before = r.adapter.context_identity()
        try:
            path = r.start_instruction(instruction)
        except (RuntimeError, ValueError, NotImplementedError) as error:
            if r.adapter.context_identity() != before or r.active is not None:
                self.faulted = True
                return dict(status='ERROR', admission='UNKNOWN', reason='START_EXCEPTION_WITH_STATE_CHANGE: '+str(error), **self.context())
            return dict(status='REJECTED', admission='NOT_ACCEPTED', tool=name,
                        reason=str(error), unchanged=True, **self.context())
        execution_id = Path(path).stem
        try:
            outcome = None
            while outcome is None:
                outcome = r.tick()
        except Exception as error:
            self.faulted = True
            if r.active:
                r.cancel()
            return dict(status='ERROR', admission='ACCEPTED', execution_id=execution_id,
                reason=type(error).__name__+': '+str(error), evidence_path=str(Path(path).resolve()), **self.context())
        data = asdict(outcome)
        self.executions[execution_id] = data
        return dict(status=outcome.status, admission='ACCEPTED', tool=name,
            execution_id=execution_id, outcome=data, source_context=before, **self.context())


def serve(bridge, incoming, outgoing):
    def emit(value):
        value = finite_data(value)
        with (bridge.runtime.evidence_dir/'bridge.jsonl').open('a', encoding='utf-8') as audit:
            audit.write(json.dumps({'direction': 'response', 'value': value}, allow_nan=False)+'\n')
        outgoing.write(json.dumps(value, allow_nan=False)+'\n')
        outgoing.flush()
    emit(bridge.ready())
    for line in incoming:
        with (bridge.runtime.evidence_dir/'bridge.jsonl').open('a', encoding='utf-8') as audit:
            audit.write(json.dumps({'direction': 'request', 'raw': line.rstrip('\r\n')})+'\n')
        try:
            request = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))
            response = bridge.handle(request)
        except (ValueError, TypeError) as error:
            response = dict(status='REJECTED', admission='NOT_ACCEPTED', reason=str(error), **bridge.context())
        except Exception as error:
            bridge.faulted = True
            response = dict(status='ERROR', session_id=bridge.runtime.session_id, reason=type(error).__name__+': '+str(error))
        emit(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', required=True, choices=['g1'])
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--precheck', action='store_true', help='Verify external inputs without creating a Runtime')
    args = parser.parse_args()
    # Keep a dedicated stdout descriptor for the protocol; third-party native/Python prints go to stderr.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    runtime = None
    try:
        with redirect_stdout(sys.stderr):
            from bodyisa.session_factory import create_session, load_config, precheck
            if args.precheck:
                protocol.write(json.dumps(precheck(load_config(args.config, args.robot)))+'\n')
                protocol.flush()
                return 0
            runtime = create_session(args.config, args.robot)
            serve(Bridge(runtime), sys.stdin, protocol)
    except Exception as error:
        protocol.write(json.dumps(dict(status='STARTUP_ERROR' if runtime is None else 'ERROR',
            reason=type(error).__name__+': '+str(error)))+'\n')
        protocol.flush()
        print(type(error).__name__+': '+str(error), file=sys.stderr)
        return 1
    finally:
        if runtime is not None:
            runtime.adapter.close()
        protocol.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
