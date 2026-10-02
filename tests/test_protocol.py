"""Protocol and injection tests without CUDA, MuJoCo, external assets or certification claims."""
import ast
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from bodyisa.bridge.jsonl import Bridge, serve, parse_request, finite_data
from bodyisa.core import StateSnapshot, CapabilityRegistry, Instruction
from bodyisa.runtime.session import ExecutionRuntime
from bodyisa.session_factory import load_profiles, load_config

ROOT = Path(__file__).resolve().parents[1]


class FakeAdapter:
    """Test apparatus only; not an implemented embodiment."""
    mass = 30
    dt = .02
    identity = {'scope': 'UNIT_TEST_ONLY', 'joint_order': ['a', 'b']}
    supported_instructions = (Instruction('STAND'),)
    def __init__(self):
        self.s = StateSnapshot(0, 0, (0, 0, .8), (1, 0, 0, 0), 0,
            (0, 0, 0), (0, 0, 0), (0, 0), (0, 0), .8, 0, (), 400, 200, False, ())
        self.capabilities = CapabilityRegistry()
        self.revision = 0
    def snapshot(self): return self.s
    def context_identity(self): return {'sim_time': self.s.sim_time, 'control_revision': self.revision}
    def close(self): pass


class ProtocolTests(unittest.TestCase):
    def test_strict_fields_and_forbidden_operands(self):
        self.assertEqual(parse_request({'tool': 'STAND'}), ('STAND', {}))
        for request in (None, [], {'tool': []}, {'tool': 'reset'}, {'tool': 'GO_TO'},
            {'tool': 'STAND', 'allowed': True}, {'tool': 'STAND', 'arguments': {'permit': True}},
            {'tool': 'STAND', 'arguments': {'joint_target': []}},
            {'tool': 'STAND', 'arguments': {'evidence_path': 'trusted.json'}},
            {'tool': 'TRACK_MOTION', 'arguments': {'target': 'dance_prefix_500', 'contract_profile': 'LEGACY'}},
            {'tool': 'get_execution', 'arguments': {'execution_id': []}}):
            with self.subTest(request=request), self.assertRaises(ValueError): parse_request(request)

    def test_persistent_jsonl_reads_and_rejection_do_not_advance(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = FakeAdapter()
            runtime = ExecutionRuntime(adapter, load_profiles()[0], directory, register_validation=False)
            bridge = Bridge(runtime)
            before = adapter.context_identity()
            incoming = StringIO('\n'.join(json.dumps(x) for x in [
                {'tool': 'describe_capabilities'}, {'tool': 'get_state'}, {'tool': 'STAND'},
                {'tool': 'get_execution', 'arguments': {'execution_id': '../other-session'}}])+ '\nnot-json\n')
            output = StringIO()
            serve(bridge, incoming, output)
            rows = [json.loads(x) for x in output.getvalue().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]['status'], 'READY')
            self.assertTrue(all(x['session_id'] == runtime.session_id for x in rows))
            self.assertEqual(rows[1]['capabilities'], [])
            self.assertEqual(rows[3]['reason'], 'NO_TRUSTED_EVIDENCE')
            self.assertEqual(rows[4]['reason'], 'UNKNOWN_EXECUTION_ID')
            self.assertEqual(rows[5]['status'], 'REJECTED')
            self.assertEqual(before, adapter.context_identity())
            self.assertIsNone(runtime.active)
            self.assertEqual(runtime.counter, 0)

    def test_unavailable_is_explicit(self):
        result = finite_data({'metric': float('nan'), 'missing': None})
        self.assertEqual(result['metric'], {'value': None, 'available': False, 'reason': 'NONFINITE'})
        self.assertIsNone(result['missing'])
        json.dumps(result, allow_nan=False)

    def test_shared_modules_do_not_import_backend_or_evaluation(self):
        for folder in ('core', 'runtime'):
            for path in (ROOT/'bodyisa'/folder).glob('*.py'):
                tree = ast.parse(path.read_text())
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import): imports.extend(x.name for x in node.names)
                    if isinstance(node, ast.ImportFrom): imports.append(node.module or '')
                self.assertFalse(any(x.startswith(('torch', 'mujoco', 'evaluation', 'bodyisa.adapters.g1', 'bfm_zero')) for x in imports), str(path))

    def test_core_and_help_do_not_load_gpu_dependencies(self):
        code = "import sys,bodyisa; from bodyisa.bridge import jsonl; assert not any(x in sys.modules for x in ['torch','mujoco']); print(bodyisa.__file__)"
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
        self.assertIn('bodyisa', result.stdout)
        result = subprocess.run([sys.executable, '-m', 'bodyisa.bridge.jsonl', '--help'], capture_output=True, text=True, check=True)
        self.assertIn('--robot', result.stdout)

    def test_profiles_packaged_and_unknown_binding_rejected(self):
        physical, semantic, entry = load_profiles()
        self.assertEqual(semantic['reference_frames'], 500)
        self.assertEqual(entry['version'], 'P32_ENTRY_V1')
        with tempfile.TemporaryDirectory() as directory:
            config = json.loads((ROOT/'configs/g1_bfm_mujoco.example.json').read_text())
            config['robot'] = 'other'
            path = Path(directory)/'config.json'; path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'UNSUPPORTED_ROBOT'): load_config(path)

    def test_test_client_and_cli_clean_jsonl_with_load_prints(self):
        # Real CLI main/OS descriptor routing with a fake, model-free Runtime.
        # This verifies transport only; it is not a physical success sample.
        import importlib.util
        spec = importlib.util.spec_from_file_location('bodyisa_test_client', ROOT/'examples/jsonl_client.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        JsonlClient = module.JsonlClient
        with tempfile.TemporaryDirectory() as directory:
            code = '''
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from test_protocol import FakeAdapter
from bodyisa.runtime.session import ExecutionRuntime
from bodyisa.session_factory import load_profiles
from bodyisa.bridge.jsonl import main
from unittest.mock import patch
runtime=ExecutionRuntime(FakeAdapter(),load_profiles()[0],sys.argv[2],register_validation=False)
def factory(*args):
    print('third-party loader diagnostic')
    return runtime
sys.argv=['bodyisa-mujoco','--robot','g1','--config','unused-by-fixture.json']
with patch('bodyisa.session_factory.create_session',side_effect=factory):
    raise SystemExit(main())
'''
            stderr_path=Path(directory)/'stderr.txt'
            with stderr_path.open('w',encoding='utf-8') as stderr:
                client=JsonlClient([sys.executable,'-c',code,str(ROOT/'tests'),str(Path(directory)/'logs')],stderr=stderr)
                first=client.request('get_state')
                second=client.request('describe_capabilities')
                self.assertEqual(client.ready['session_id'],first['session_id'])
                self.assertEqual(first['session_id'],second['session_id'])
                self.assertEqual(first['context_identity'],second['context_identity'])
                self.assertEqual(client.request('STAND')['status'],'REJECTED')
                client.close()
                self.assertEqual(client.process.returncode,0)
            self.assertIn('third-party loader diagnostic',stderr_path.read_text())
