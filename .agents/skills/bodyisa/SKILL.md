---
name: bodyisa
description: Query and operate the current project's G1 body through its persistent BodyISA JSONL bridge in offline BFM-Zero MuJoCo simulation. Use for capability/state queries and user-requested registered STAND or TRACK actions; not hardware or benchmark execution.
---

Use this skill as Codex's client for BodyISA. G1Adapter currently binds G1,
BFM-Zero and MuJoCo. Prepare the explicit local config and external assets/evidence
using [docs/ASSETS.md](../../../docs/ASSETS.md) (path relative to the repository
root is `docs/ASSETS.md`). Do not regenerate approval to make an action start.

Start once from the project environment:

```text
bodyisa-mujoco --robot g1 --config configs/g1_bfm_mujoco.local.json
```

Equivalent: `python -m bodyisa.bridge.jsonl --robot g1 --config <local-config>`.
Use a persistent exec terminal (`tty=true` on the tested Windows client), retain
its session id, and send newline-terminated JSON using `write_stdin` to the same
session. Read `status=READY` before dispatch; startup errors mean no ready robot.
Model output is stderr. PTY echo is not a response; actual protocol is also saved
to the local output directory's `bridge.jsonl`. Do not restart per request.

Before operating, send:

```jsonl
{"tool":"describe_capabilities"}
{"tool":"get_state"}
```

For a query-only user request, stop after reporting these results. Choose each
subsequent action from the user's goal and current `entry_eligible`/reason.
The skill does not impose a task sequence; the Runtime enforces its reviewed
single-chain context and requires an actual preceding STAND for TRACK.

The complete supported action/query argument forms are:

```jsonl
{"tool":"STAND"}
{"tool":"TRACK_MOTION","arguments":{"target":"dance_prefix_500","contract_profile":"JOINT_TRAJECTORY_V2"}}
{"tool":"get_execution","arguments":{"execution_id":"002_TRACK_MOTION"}}
```

Send one instruction and await its real terminal response before deciding the
next step. `admission=ACCEPTED` is not SUCCESS; check `outcome.status`, failure
reason, semantic assessment, exit result, state, phase, eligibility and evidence
path. Preserve failed outcomes. Null/unavailable observations are not zero.
Only session-local returned execution ids can be queried.

Codex wall-clock waits do not advance physics or stable hold. Keep the existing
session for conversational follow-ups. CLOSED permits queries, but not repeated
TRACK; report the limitation instead of resetting. Unsupported navigation,
arbitrary joint targets/motions and hardware cannot be emulated through this
interface. Do not edit thresholds, evidence, gains, latent or control internals
to bypass rejection, and do not send permit/reset/evidence fields.

This is a synchronous simulation interface, not real-time hardware control. The
Skill is usage guidance, not a security sandbox. Full protocol and configuration
details: repository `docs/CODEX_USAGE.md` and `README.md`.
