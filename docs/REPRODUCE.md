# Install, load reviewed deployment, and reproduce one smoke

Normal use: install the package/G1 dependencies, obtain the exact external inputs
and migration deployment directory from the maintainer, fill the local config,
and start the bridge. Source and profiles must match the deployment digests.
No approval or benchmark is run during normal startup.

## Explicit maintainer migration review

Keep the original Phase 3.2 workspace/evidence external and unchanged. Obtain the
exact files in `evaluation/baseline_inventory.json`; if starting from the original
workspace, export them to a new private directory:

```powershell
python scripts/export_baseline.py --baseline-root C:\external\original-bodyisa --output C:\external\bodyisa-baseline-export
```

Set up the G1 configuration and external assets first. Run all regressions from
this checkout using its installed environment, retaining the full output. Then
run the explicitly separate local review command:

```powershell
$env:BODYISA_BFM_ROOT = 'C:\external\BFM-Zero-minimal'
python -m unittest discover -s tests -v 2> regression.txt
python -m evaluation.migration --baseline-root C:\external\bodyisa-baseline-export --config configs/g1_bfm_mujoco.local.json --regression-log regression.txt --output C:\external\bodyisa-deployment-new
```

The output must not exist. Only after baseline/hash/AST/profile/regression and
actual loaded backend checks pass does it write a new `LOCAL_AUTOMATED_REVIEW`,
new bundles and independent `trust.json`. Point the local config's evidence fields
to that output. This is a visible maintenance action, not an Agent instruction or
an automatic startup workaround. A version/asset mismatch remains a failure;
never rewrite old digests. See MIGRATION for the exact mapping and scope.

## One ordinary persistent-process smoke

Use a new output directory in the config; startup refuses to overwrite one.
The migration review freezes `smoke_plan.json` before this run: seed 91000,
one episode; three actual successes/500 semantic samples/exit SUCCESS/CLOSED;
TRACK metric comparison absolute tolerance 1e-5 rad, consecutive-duration
1e-9 s, per-stage elapsed time 0.04 s, root displacement and heading 0.001 m/rad.
These are diagnostic migration comparison tolerances, not changed control
contracts or new capability boundaries.

Start the actual installed bridge:

```text
bodyisa-mujoco --robot g1 --config configs/g1_bfm_mujoco.local.json
```

Wait for READY, then send and inspect each response in the same process:

```jsonl
{"tool":"describe_capabilities"}
{"tool":"get_state"}
{"tool":"STAND"}
{"tool":"describe_capabilities"}
{"tool":"TRACK_MOTION","arguments":{"target":"dance_prefix_500","contract_profile":"JOINT_TRAJECTORY_V2"}}
{"tool":"STAND"}
{"tool":"get_state"}
{"tool":"TRACK_MOTION","arguments":{"target":"dance_prefix_500","contract_profile":"JOINT_TRAJECTORY_V2"}}
```

The last request must be refused without a control step. Queries do not advance
sim time. End stdin to close the process; an exit code cannot replace physical
outcomes. The same `Bridge` is available to the subprocess test client in examples.

Read-only analysis (safe to repeat; does not modify existing logs):

```text
python -m evaluation.smoke --run-dir runs/g1-session-001 --baseline-root C:/external/bodyisa-baseline-export --deployment-root C:/external/bodyisa-deployment-new
```

Use `--output NEW_FILE.json` only to save a new derived analysis. Missing necessary
records return UNVERIFIABLE. The checker inspects original execution, admission,
setup and bridge logs, source/entry continuity, no permit, invariants, semantic
and exit results, queries/CLOSED rejection and baseline diagnostics. Old evidence
is neither rewritten nor claimed to execute the new code.

## Packaging and basic checks

```text
python -m pip wheel --no-deps --no-build-isolation . --wheel-dir dist
python -m unittest discover -s tests -p test_protocol.py -v
```

Wheel and source distributions include profiles. Base tests and `--help` do not
need torch, MuJoCo or CUDA. Full regressions need explicit BFM source and G1
simulation assets. Physical smoke additionally needs the checkpoint/caches and
reviewed evidence. Missing GPU/assets are SKIPPED/BLOCKED conditions, never
reported as physical PASS. Local validation used a separate installation and
left the original environment unchanged.
