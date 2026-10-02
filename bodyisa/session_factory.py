"""Local setup only: external assets -> concrete Adapter -> ordinary persistent Runtime."""
from pathlib import Path
from importlib.resources import files
import json
from bodyisa.runtime.evidence import file_hash

PATH_KEYS = ('bfm_root', 'model_dir', 'motion', 'track_latent', 'stand_latent',
             'evidence_root', 'trust_anchor', 'stand_bundle', 'track_bundle', 'output_dir')


def load_config(path, robot='g1'):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    expected = {'robot', 'controller', 'execution_backend', 'device', 'seed', *PATH_KEYS}
    if set(config) != expected:
        raise ValueError(f'CONFIG_FIELDS: missing={sorted(expected-set(config))}, unknown={sorted(set(config)-expected)}')
    if robot != 'g1' or config['robot'] != 'g1' or config['controller'] != 'BFM-Zero' or config['execution_backend'] != 'MuJoCo':
        raise ValueError('UNSUPPORTED_ROBOT_CONTROLLER_BACKEND')
    if config['device'] != 'cuda' or type(config['seed']) is not int or config['seed'] != 91000:
        raise ValueError('DEPLOYMENT_SCOPE: current migration smoke uses CUDA and seed 91000')
    for key in PATH_KEYS:
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError('CONFIG_PATH_REQUIRED: '+key)
        config[key] = str((path.parent / config[key]).resolve())
    return config


def load_profiles():
    root = files('bodyisa.profiles.g1')
    return tuple(json.loads(root.joinpath(name).read_text(encoding='utf-8'))
                 for name in ('physical.json', 'track_v2.json', 'entry_policy.json'))


def precheck(config):
    """No simulator/model initialization; strict explicit files and byte identities."""
    manifest = json.loads(files('bodyisa.profiles.g1').joinpath('external_manifest.json').read_text())
    root = Path(config['bfm_root'])
    for relative, expected in manifest['bfm_python_sources'].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError('EXTERNAL_BFM_SOURCE_MISSING: '+str(path))
        if file_hash(path) != expected:
            raise RuntimeError('EXTERNAL_BFM_SOURCE_HASH_MISMATCH: '+relative)
    paths = {**config, 'checkpoint': str(Path(config['model_dir'])/'model.safetensors'),
             'model_config': str(Path(config['model_dir'])/'config.json'),
             'model_init': str(Path(config['model_dir'])/'init_kwargs.json'),
             'xml': str(root/'bfm_zero_inference_code/g1_for_reward_inference.xml')}
    for key, expected in manifest['files'].items():
        path = Path(paths[key])
        if not path.is_file():
            raise FileNotFoundError('EXTERNAL_ASSET_MISSING: '+str(path))
        if file_hash(path) != expected:
            raise RuntimeError('EXTERNAL_ASSET_HASH_MISMATCH: '+key)
    return {'status': 'INPUTS_VERIFIED', 'source_files': len(manifest['bfm_python_sources'])}


def build_adapter(config):
    precheck(config)
    from .adapters.g1.config import configure_external
    configure_external(config['bfm_root'])
    from .adapters.g1.adapter import G1Adapter, ReferenceMotion, DEFAULT_JOINT_POS, make_env
    from bfm_zero_inference_code.fb_cpr_aux.model import FBcprAuxModel
    import torch
    import numpy as np
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = FBcprAuxModel.load(config['model_dir'], device=config['device'], strict=True)
    adapter = G1Adapter(model, config['device'], paths=config)
    try:
        # Same frozen Phase 2 caches and installation operations; no inference re-encoding.
        stand_path = Path(config['stand_latent'])
        with np.load(stand_path) as data:
            target, z = data['joint_target'].copy(), data['z'].copy()
        np.testing.assert_array_equal(target, DEFAULT_JOINT_POS)
        adapter.poses.register('default_pose', 'external_bfm/common.py:DEFAULT_JOINT_POS',
                               target, str(stand_path), torch.tensor(z, device=adapter.device))
        for name in ('STAND', 'REACH_POSE'):
            adapter.capabilities.entries[name]['evidence_levels'].append('ASSET_AVAILABLE')
        adapter.reference = ReferenceMotion(config['motion'], 500)
        track_path = Path(config['track_latent'])
        with np.load(track_path) as data:
            latent, times, fps = data['z'].copy(), data['times'].copy(), float(data['source_fps'])
        if latent.shape != (500, 256) or fps != adapter.reference.fps:
            raise RuntimeError('TRACK_CACHE_SHAPE_OR_FPS_MISMATCH')
        np.testing.assert_allclose(times, np.arange(len(times))*adapter.dt, atol=1e-7)
        adapter.track_z = torch.tensor(latent, device=adapter.device)
        adapter.metric_env = make_env()
        adapter.track_source = dict(mechanism='tracking_inference', cache=str(track_path),
            cache_sha256=file_hash(track_path), reference_path=adapter.reference.path,
            reference_sha256=adapter.reference.source_hash, reference_id=adapter.reference.reference_id,
            source_fps=adapter.reference.fps, frames=len(adapter.reference.joints),
            duration_s=adapter.reference.duration, latent_period_s=adapter.dt,
            body_metric_source='FK_DERIVED_FROM_REFERENCE_JOINTS_AND_ROOT', reference_alignment='FIXED_WORLD_V1')
        adapter.capabilities.entries['TRACK_MOTION']['evidence_levels'].append('ASSET_AVAILABLE')
        adapter.capabilities.require_envelope = True
        return adapter
    except BaseException:
        adapter.close()
        raise


def create_session(config_path, robot='g1'):
    config = load_config(config_path, robot)
    output = Path(config['output_dir'])
    if output.exists():
        raise FileExistsError('OUTPUT_ALREADY_EXISTS: '+str(output))
    output.mkdir(parents=True)
    from .runtime.session import ExecutionRuntime
    import numpy as np
    adapter = build_adapter(config)
    try:
        physical, semantic, entry = load_profiles()
        runtime = ExecutionRuntime(adapter, physical, output, register_validation=False,
            track_v2_config=semantic, entry_policy=entry,
            allowed_evidence_root=config['evidence_root'], trusted_registry_path=config['trust_anchor'])
        delta = np.random.default_rng(config['seed']).uniform(-.001, .001, len(adapter.identity['joint_order']))
        runtime.initialize_episode('g1-'+runtime.session_id, delta)
        runtime.load_evidence(config['stand_bundle'])
        runtime.load_evidence(config['track_bundle'])
        (output/'setup.json').write_text(json.dumps(dict(config=config, seed=config['seed'],
            joint_delta_rad=delta.tolist(), session_id=runtime.session_id, evaluation_permit=None,
            imported_modules={'bodyisa': str(Path(__file__).resolve().parent),
                              'factory': str(Path(__file__).resolve())},
            actual_identity=runtime._backend_identity), indent=2), encoding='utf-8')
        return runtime
    except BaseException:
        adapter.close()
        raise
