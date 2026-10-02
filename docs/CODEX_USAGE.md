# Codex client workflow

The repository skill is `.agents/skills/bodyisa/SKILL.md`, with `name: bodyisa`.
See the [official skill documentation](https://learn.chatgpt.com/docs/build-skills)
for skill discovery. Open this repository as the workspace and request:

> $bodyisa 查询当前 G1 仿真的能力，不要执行动作。

Set up assets and a reviewed deployment as described in ASSETS first. Start
`bodyisa-mujoco --robot g1 --config configs/g1_bfm_mujoco.local.json` once. In
Codex's terminal tools, retain the running exec session id and write JSON lines
using `write_stdin` to that same session. A PTY (`tty=true`) was tested on the
current Windows client across separate calls. PTY input echo/control sequences
are terminal presentation, not protocol output; `runs/.../bridge.jsonl` records
actual protocol requests/responses. A subprocess with pipes receives clean JSONL.

Read READY before sending a request. Before an action, query capabilities and
state; inspect current `entry_eligible` and reason. Request only the action that
the user wants. Each action response is terminal, because local ticks run
synchronously until the real contract terminates. Check outcome, not just
admission or console exit code. Retrieve past results with a session-local
execution id; filesystem paths are not accepted as execution ids.

Waits between Codex calls do not advance simulation. Reuse the same process for
follow-ups such as “先站稳”, “演示当前注册的舞蹈片段”, “最后站稳”. Do not use a fresh
`echo | python` process per action. CLOSED allows queries and rejects further
motion; starting another episode is a new explicit task, not automatic recovery.

The test client in `examples/jsonl_client.py` and actual Codex terminal requests
are different forms of validation. Their observed results are recorded separately
in `docs/experiments/RESULTS.md`. Neither is a new GPT API or MCP server.
