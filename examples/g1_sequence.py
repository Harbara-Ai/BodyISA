"""Explicit ordinary API example; each result is checked before the next action."""
import argparse
import json
from bodyisa.bridge.jsonl import Bridge
from bodyisa.session_factory import create_session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    runtime = create_session(args.config, 'g1')
    bridge = Bridge(runtime)
    try:
        for request in ({'tool': 'STAND'}, {'tool': 'TRACK_MOTION', 'arguments': {
                'target': 'dance_prefix_500', 'contract_profile': 'JOINT_TRAJECTORY_V2'}}, {'tool': 'STAND'}):
            result = bridge.handle(request)
            print(json.dumps(result), flush=True)
            if result['status'] != 'SUCCESS':
                return 1
        return 0
    finally:
        runtime.adapter.close()


if __name__ == '__main__':
    raise SystemExit(main())
