# Third-party sources and publication scope

- BFM-Zero: [LeCAR-Lab/BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero), minimal
  inference branch. The local reference snapshot was obtained as ZIP, without a
  Git revision. `bodyisa/profiles/g1/external_manifest.json` records the 14 actual
  Python source hashes used by this backend. Networks, normalization, actor,
  environment and PD remain external and are not vendored here.
- The existing BFM source tree declares CC BY-NC 4.0. Its unmodified supplied
  license is retained at [licenses/BFM-Zero-LICENSE.txt](licenses/BFM-Zero-LICENSE.txt).
  Upstream model code includes Meta Platforms copyright notices. Keep all
  upstream notices with externally obtained code. Model, motion and mesh rights
  must be checked at their respective sources; this repository does not grant
  redistribution rights for them.
- [Harbara-Ai/DexISA_v2](https://github.com/Harbara-Ai/DexISA_v2) was inspected
  for responsibility boundaries, Protocol, factory, persistent JSONL and Skill
  organization. Its controller/robot code and assets were not copied.
- MuJoCo, PyTorch, NumPy, SciPy and other installed libraries retain their own
  licenses; dependencies are declared, not copied into the repository.

The license for BodyISA's own middleware is pending an explicit owner decision.
No BFM or DexISA license has been assigned to all BodyISA code. Publication of
the source is not represented as a new license grant.
