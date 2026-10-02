# External sources, assets and evidence

No model, mesh, motion, latent archive, virtual environment, raw batch rollout or
video is distributed in this Git repository. No startup command downloads them.

| Required input | Source / role | Identity |
|---|---|---|
| BFM minimal source tree | [official minimal inference branch](https://github.com/LeCAR-Lab/BFM-Zero/tree/minimal_inference); `env.py`, `common.py`, model modules, XML and referenced meshes | 14 Python hashes + XML hash in packaged external manifest |
| Model directory | [official model repository](https://huggingface.co/LeCAR-Lab/BFM-Zero); `model.safetensors`, `config.json`, `init_kwargs.json` | checkpoint `d8d07d7b83ad030286f44d55a48899dc42db4b48000132fffad063653a6eb49e` |
| Registered motion | upstream `example_motion/dance1_subject2_50_jpos.npz` | `48fc271ad378e397f70cde265ccbe91e56c16ef4b889945ed76eb71e3b87ca17` |
| Frozen TRACK latent | Phase 2 reviewed `dance_prefix_latents.npz`, supplied by BodyISA maintainer | `23756e2bf8d9d137dd37840c06ad65eba9ec54b82df076b6258192eb0d7215ad` |
| Frozen STAND latent | Phase 2 reviewed `default_pose_latent.npz`, supplied by BodyISA maintainer | `84eec1f906acbb8ecc66dbf55d2dd3c7eaa9e6c22d8b6c6b71c8130e1caab16b` |
| Deployment evidence directory | maintainer-reviewed migration export: `trust.json`, two bundles, review, compatibility and exact historical payloads | pinned bundle digests; full loaded backend identity compared at startup |

The public manifest is [external_manifest.json](../bodyisa/profiles/g1/external_manifest.json).
Upstream is mutable: a latest checkout is not claimed compatible with this
ZIP-derived snapshot. A differing source must receive a new explicit review;
do not update expected hashes merely to start it. No upstream commit is invented.
Follow source-specific licenses; the retained BFM license is CC BY-NC 4.0.

Copy `configs/g1_bfm_mujoco.example.json` to an ignored `.local.json` and fill in
`bfm_root`, `model_dir`, `motion`, `track_latent`, `stand_latent`, `evidence_root`,
`trust_anchor`, `stand_bundle`, `track_bundle`, and a fresh `output_dir`.
Paths resolve relative to the config file. All bundle payloads are resolved
under the configured evidence root and checked by content hash. Source code is
imported only from the installed BodyISA package and explicitly supplied BFM
tree. There is no implicit import from the old BodyISA workspace.

Maintainer-only migration/reproduction needs the original Phase 3.2 evidence
export, indexed by [baseline_inventory.json](../evaluation/baseline_inventory.json).
`scripts/export_baseline.py` creates a byte-exact external directory from the
retained original workspace. It contains private raw paths and is **not** a
sanitized public report. Request this export and the exact reviewed latent
caches from the repository maintainer if they are not already available; there
is no fictitious public download URL. Alternatively obtain the already-approved
migration deployment directory from that maintainer and configure it directly.
Maintainers can reproduce it using [REPRODUCE](REPRODUCE.md). Clone alone is
insufficient for physical startup.

Missing source/assets produce `EXTERNAL_BFM_SOURCE_MISSING` /
`EXTERNAL_ASSET_MISSING`. Changed bytes produce a hash mismatch. Missing or
mismatched deployment evidence fails startup before READY or any instruction.
Ordinary execution cannot reapprove itself. The dependency/asset precheck is:

```text
bodyisa-mujoco --robot g1 --config configs/g1_bfm_mujoco.local.json --precheck
```

Precheck checks files/hashes only; READY additionally requires real model,
simulator and evidence identity verification. Neither runs a benchmark.
