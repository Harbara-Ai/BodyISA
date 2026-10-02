# Phase 3.2 package migration

The original passed workspace is retained unchanged, with a 152-file baseline
inventory (source/config/reports and exact final referenced evidence). Public
`evaluation/baseline_inventory.json` contains relative names, byte sizes and
hashes. It is an index, not a substitute for the externally retained bytes.

| Previous implementation | Current implementation | Nature |
|---|---|---|
| `bodyisa/core.py` dataclasses | `bodyisa/core/{instruction,state,outcome}.py` | mechanical extraction |
| physical/transition contracts | `bodyisa/core/contracts.py` | same class/method AST |
| capability/pose types | `bodyisa/core/capability.py` | same existing registry; verified bundle registration retained |
| ExecutionRuntime | `bodyisa/runtime/session.py` | lifecycle retained; initialization/observer/attestation injected through Adapter |
| `bodyisa/admission.py` | `bodyisa/runtime/admission.py` | byte-preserved implementation, old import forwards |
| data-only evidence verification | `bodyisa/runtime/evidence.py` | pinned trust/payload checks retained; capability comes from attested catalog |
| actual G1 backend attestation | `bodyisa/adapters/g1/identity.py` | G1-only; explicit external paths, expanded source coverage |
| `BFMAdapter` | `bodyisa/adapters/g1/adapter.py:G1Adapter` | same physics/observation/action/reference methods; local setup extracted |
| `bodyisa/track_contract_v2.py` | `bodyisa/profiles/g1/track_contract_v2.py` | byte-preserved observer; old import forwards |
| physical/v2/entry JSON | `bodyisa/profiles/g1/*.json` | byte-preserved files, no numeric changes |
| phase3 helpers / phase31 audit and manifest | `evaluation/` | retained consumers; excluded from normal setup path |
| phase32 example setup | `session_factory.py`, `bridge/jsonl.py` | explicit external configuration and new persistent protocol |

`bodyisa.core` re-exports former types and lazily exports ExecutionRuntime and
local evaluation permit helpers for historical regression consumers.
`bodyisa.bfm_adapter.BFMAdapter` aliases the one G1Adapter implementation; it
does not maintain a second controller. Historical reports/source archives are
not globally renamed. Tests adapt import/resource paths and injected observer
interfaces; all original 37 assertions/tests remain.

The original bundles certify the original source hashes. They are not accepted
as new deployment evidence. `python -m evaluation.migration` explicitly:

1. verifies the pinned baseline inventory, original final anchor and bundles;
2. compares original/current contract, tick, termination, state/action/reference
   and entry/receipt ASTs and exact profile bytes;
3. requires a complete passing regression log;
4. constructs the current G1 backend and requires every old non-source
   physical/semantic identity to match exactly;
5. emits a separate local automated migration review, copied byte-exact
   historical payloads, new bundle digests and a new independent trust anchor.

The mapping records old/new source hashes, method-level comparisons and the
responsibilities that changed. It does not use a bare `compatibility=true`.
Actual startup separately rechecks model tensors, compiled simulator, PD,
reference, latents, all package Python source and 14 external BFM source hashes.
The new JSONL smoke then checks the new package, without backfilling old reports.

`.gitattributes` disables newline conversion so checkout and wheel preserve
the bytes used in identity checks. Configured evidence roots and relative
payload paths permit relocation; filenames alone never establish identity.
Public experiment summaries are derived and labelled. Raw execution logs and
their private absolute paths remain external and byte-exact.
