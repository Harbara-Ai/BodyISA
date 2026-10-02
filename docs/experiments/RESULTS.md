# Published derived experiment summary

These are public summaries and content-hash indexes, **not original execution
logs**. Raw reports, paths, histories and rollouts remain in the separately
retained local evidence exports described in [ASSETS](../ASSETS.md).

## Original execution versions

| Evidence | Meaning | Observed result |
|---|---|---|
| Phase 3 | legacy reference-consumed + terminal stabilization, not a semantic dance contract | stand-latent negative controls also passed, so not v2 support |
| Phase 3.1 D2 | new JOINT_TRAJECTORY_V2 semantic scoring | normal 56/56 TRACK PASS; three negative-control types 24/24 semantic FAIL, despite 24 successful exits |
| Phase 3.2 final A | evaluation permit bypassed only candidate certification | 8/8 complete chains |
| Phase 3.2 B | independent seeds, ordinary API without permit | 8/8 complete chains, 24 external instruction SUCCESS; 32 expected refusals without control effects |
| 2026-10-01 final-bundle smoke | original final anchor/bundles through original sequence CLI | one real complete chain, 500-sample semantic PASS, CLOSED |

Original run ids: `p31_d2_v2_20260930b`, `p32_a_v1b_20260930`,
`p32_b_v1_20260930`, `p32_final_smoke_20261001_01`. The first A run's review-tool
syntax error and fixed-seed repeat are retained in the original evidence; they
are not counted as another independent validation group.

Source indexes: [baseline_inventory.json](../../evaluation/baseline_inventory.json)
and external original `BODYISA_PHASE31_RESULTS.md`, `BODYISA_PHASE32_RESULTS.md`,
`BODYISA_FINAL_SMOKE_RESULT.md`. These named original reports are external, not
missing repository links.

## 2026-10-02 modular package / persistent Bridge smoke

**SMOKE_PASS** in one installed `bodyisa-mujoco` process, outside the source
checkout working directory. Runtime had no evaluation permit; the migrated
reviewed deployment loaded successfully. The **current Codex** sent actual
JSONL requests using `write_stdin` across separate terminal calls, retained one
session, inspected intermediate outcomes and continued through the requested
chain. This was not a pre-scripted client being labelled as natural-language
Agent testing. Automatic Skill discovery in a fresh chat was not separately
tested.

| Execution | Ordinary admission | Actual outcome | Simulation time |
|---|---|---|---:|
| 001_STAND | ACCEPTED | SUCCESS | 0.86 s |
| 002_TRACK_MOTION | ACCEPTED | SUCCESS; 500 semantic samples / PASS; exit SUCCESS | 11.10 s |
| 003_STAND | ACCEPTED | SUCCESS; CLOSED | 0.52 s |

Two external handoffs retained exactly equal state/context/history/action
identities. All 624 physics steps had no invariant violation. There was one
episode initialization, no chain reset/teleport/history clear. Queries left
state unchanged; CLOSED TRACK returned UNSUPPORTED_TRACK_CONTEXT without a
control step. A post-CLOSED `get_execution` returned the actual previous TRACK
outcome. CLI ended with stdin EOF, exit code 0; stderr was empty.

TRACK mean joint RMSE **0.108013156580 rad**, active-group mean
**0.121456679366 rad**, rolling25 maximum **0.235367203338 rad**, longest
consecutive excess **0.14 s**. Complete-chain root displacement
**0.181245329032 m**, absolute heading change **0.294253833407 rad**.
All values match the original seed-91000 final smoke within the predeclared
tolerances (in this run, the compared numbers are identical). These diagnostics
do not establish zero drift or world-path accuracy.

Evidence: [migration_smoke_summary.json](migration_smoke_summary.json) records
41 direct raw-log checks, tolerances, outcomes, handoffs and raw-file hashes;
[deployment_identity.json](deployment_identity.json) records actual loaded
identities; [migration_mapping.json](migration_mapping.json) records 13 identical
AST comparisons and old/new hashes. Raw run `bridge_smoke_01` is local/external.

The original final trust/bundle files were not edited. A distinct explicit
`BODYISA_032_MODULE_MIGRATION_REVIEW_V1` created new deployment evidence from
verified baseline bytes, compatibility tests and equal physical identities.
Its trust/bundle digests are included in the summary. The new smoke validates
the migrated version only and does not enlarge its capability/initial-state scope.

## Tests and installation

- Baseline: original **37/37** regressions PASS before migration.
- Migrated package: all original **37** tests retained with import/resource and
  injected-interface adaptations; no assertion removed.
- With added protocol/injection/install tests: **44/44 PASS**.
- The JSONL **test client** separately passed a persistent subprocess test of the
  actual CLI transport using a model-free fixture. Loader prints reached stderr,
  stdout contained only parseable JSONL, queries preserved state and unapproved
  actions were refused. This fixture is not physical success evidence.
- The **current Codex** independently completed the real physical Bridge smoke
  described above. No local forwarding daemon was necessary; a persistent PTY
  was usable. Plain non-PTY exec closed stdin on this client and was not used.
- Wheel installation, console entry and `python -m ... --help` were checked from
  a different directory; module `__file__` paths came from the new environment's
  installed wheel. The original environment was not modified. Local dependency
  binaries were explicitly reused from its site-packages through a new-environment
  `.pth` for validation; this is not an implicit old-BodyISA import or a claim of
  testing a fresh internet dependency download.
- Skill frontmatter validation passed. Contract/entry resources are retained
  byte-exact and included in wheel/source distributions.

The clean Git checkout/install and release-content checks are recorded in
[release_checks.json](release_checks.json). GPU smoke was performed once as
above, not repeated as a new benchmark in the clean checkout.
