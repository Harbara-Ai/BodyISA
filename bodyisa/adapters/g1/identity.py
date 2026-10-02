"""G1-only loaded actor, simulator, asset, preprocessing and source attestation."""
from pathlib import Path
import numpy as np
from bodyisa.runtime.evidence import array_hash,digest,file_hash

def source_identity(bfm_root):
    package=Path(__file__).resolve().parents[2]
    result={str(p.relative_to(package.parent)).replace('\\','/'):file_hash(p)
            for p in sorted(package.rglob('*.py'))}
    external=Path(bfm_root)
    result.update({'external_bfm/'+str(p.relative_to(external)).replace('\\','/'):file_hash(p)
                   for p in sorted(external.rglob('*.py')) if '__pycache__' not in p.parts})
    return result

def actual_identity(adapter,physical,v2,entry_policy):
    """Called at setup/import, never in the control loop. Claims come from loaded objects."""
    import mujoco
    from safetensors import safe_open
    from .adapter import DEFAULT_JOINT_POS
    if adapter.model is None:raise RuntimeError('BACKEND_MODEL_NOT_LOADED')
    model_dir=Path(adapter.paths['model_dir'])
    # Bind the actual actor and observation normalizer to checkpoint tensors.
    if not hasattr(adapter,'_verified_actor_digest'):
        loaded=adapter.model.state_dict()
        keys=sorted(k for k in loaded if k.startswith(('_actor.','_obs_normalizer.')))
        if not keys:raise RuntimeError('BACKEND_ACTOR_IDENTITY_UNAVAILABLE')
        pairs={}
        with safe_open(model_dir/'model.safetensors',framework='pt',device='cpu') as source:
            for key in keys:
                actual=array_hash(loaded[key]);expected=array_hash(source.get_tensor(key))
                if actual!=expected:raise RuntimeError('LOADED_MODEL_CHECKPOINT_MISMATCH:'+key)
                pairs[key]=actual
        adapter._verified_actor_digest=digest(pairs)
        adapter._verified_checkpoint_digest=file_hash(model_dir/'model.safetensors')
    ref=adapter.reference
    if ref is None or ref.reference_id!='dance_prefix_500':raise RuntimeError('ASSET_NOT_REGISTERED')
    with np.load(ref.path) as source:
        if not np.array_equal(ref.joints,source['dof_positions'][:500]):
            raise RuntimeError('LOADED_REFERENCE_MISMATCH')
        if not np.array_equal(ref.positions,source['body_positions'][:500]) or not np.array_equal(ref.quaternions,source['body_rotations'][:500]):
            raise RuntimeError('LOADED_REFERENCE_PREPROCESSING_MISMATCH')
    track_path=Path(adapter.paths['track_latent'])
    stand_path=Path(adapter.paths['stand_latent'])
    with np.load(track_path) as source:
        if array_hash(adapter.track_z)!=array_hash(source['z']):raise RuntimeError('LOADED_TRACK_LATENT_MISMATCH')
    with np.load(stand_path) as source:
        pose=adapter.poses.get('default_pose')
        if array_hash(pose['latent'])!=array_hash(source['z']) or not np.array_equal(pose['joint_target'],DEFAULT_JOINT_POS):
            raise RuntimeError('LOADED_STAND_LATENT_MISMATCH')
    compiled=np.empty(mujoco.mj_sizeModel(adapter.env.mjm),dtype=np.uint8)
    mujoco.mj_saveModel(adapter.env.mjm,None,compiled)
    controller={key:array_hash(getattr(adapter.env,key)) for key in
        ('kp_gains','kd_gains','default_joint_pos','action_scales')}
    controller.update(action_rescale=adapter.env.action_rescale,sim_dt=adapter.env.sim_dt,
                      n_substeps=adapter.env.n_substeps)
    return dict(checkpoint_sha256=adapter._verified_checkpoint_digest,
        actual_actor_normalizer_sha256=adapter._verified_actor_digest,
        model_config_sha256=file_hash(model_dir/'config.json'),
        model_init_kwargs_sha256=file_hash(model_dir/'init_kwargs.json'),
        simulator_version=mujoco.__version__,compiled_model_sha256=array_hash(compiled),
        simulator_xml_sha256=file_hash(Path(adapter.paths['bfm_root'])/'bfm_zero_inference_code/g1_for_reward_inference.xml'),
        controller_parameters_sha256=digest(controller),control_dt_s=adapter.dt,
        asset_sha256=file_hash(ref.path),reference_joints_sha256=array_hash(ref.joints),
        reference_preprocessing='FIXED_WORLD_V1',track_latent_sha256=file_hash(track_path),
        stand_latent_sha256=file_hash(stand_path),physical_contract_hash=digest(physical),
        semantic_contract_hash=digest(v2),entry_policy_hash=digest(entry_policy),
        source_sha256=source_identity(adapter.paths['bfm_root']),
        capabilities=adapter.capability_catalog,
        embodiment='G1',controller='BFM-Zero',execution_backend='MuJoCo')
