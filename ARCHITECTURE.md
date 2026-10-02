# BodyISA responsibilities

Codex → persistent JSONL Bridge → injected ExecutionRuntime → BodyAdapter → G1Adapter.
`session_factory` operates outside dispatch: explicit configuration, external
sources/assets, profiles, controlled initialization and trusted bundle loading.
One factory call creates one model, live Adapter and Runtime. Separate scratch
and reference MuJoCo data used by the retained snapshot/metric code do not
control the live episode.

| Layer | Implementation | Responsibility |
|---|---|---|
| Shared types | `bodyisa/core/` | Instructions, canonical states, outcomes, physical/transition contracts, capability evidence |
| Shared Runtime | `bodyisa/runtime/session.py` | Ownership, authorization, entry checks, receipts, tick/termination, failure latch |
| Entry/evidence | `bodyisa/runtime/admission.py`, `evidence.py` | State neighborhoods, opaque context equality, independently pinned data-only bundle loading |
| Adapter contract | `bodyisa/adapters/base.py` | Actual Runtime calls; state, realization, clock, context, identity and resource lifetime |
| G1 implementation | `bodyisa/adapters/g1/` | G1 joint/support mappings, BFM observations/actor/latent, MuJoCo reads and actual loaded identity |
| G1 profile | `bodyisa/profiles/g1/` | Byte-preserved physical/v2/entry configs and state-only 500-sample observer |
| Factory | `bodyisa/session_factory.py` | Explicit G1/BFM-Zero/MuJoCo binding and normal local setup |
| Bridge | `bodyisa/bridge/jsonl.py` | Strict requests, one instruction per action, persistent session, structured results |
| Evaluation | `evaluation/` | Offline compatibility/review and read-only smoke analysis; no normal startup dependency |

Shared core/runtime import no concrete G1, BFM, torch or MuJoCo implementation.
Initialization is delegated to the Adapter; the profile observer is supplied
through the Adapter. Runtime consumes normalized state and opaque context
identities rather than understanding BFM history arrays or latent dimensions.
The retained legacy capability evidence fields exist for compatibility, but
cannot authorize ordinary deployment.

State uses world-frame root position/linear/angular velocity, wxyz quaternion,
radians and seconds. G1 joint names are supplied by READY/backend identity;
state vectors follow that order. Actual contact/support measurements retain
their provenance. Nonfinite protocol values are explicit null/unavailable
records, never zero; missing optional assessments stay null.

Admission is evaluated again at start. Evidence validity, static authorization
and current eligibility are distinct. Only actual completion produces a receipt.
The initial STAND and post-TRACK explicit STAND have different entry contexts.
TRACK internal stabilization cannot substitute for the external instruction.
Source terminal and next entry must match integration/history/action/revision.

Semantic failures remain failed after successful stabilization. Waiting between
requests never calls tick. Bridge exceptions latch a fault and cannot restart
the episode. Local configuration/review is trusted setup; ordinary JSONL requests
cannot select evidence, edit thresholds or obtain evaluation permits.

The local trust model assumes control over the Python process and trust files.
It checks explicit review, integrity and scope, not hostile process memory.
No PKI, asynchronous scheduler, transport servo loop or hardware backend exists.
