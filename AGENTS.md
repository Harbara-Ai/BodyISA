# BodyISA workspace

Preserve the G1 BFM-Zero MuJoCo physical contracts, source attestations and frozen
evidence. Normal sessions have no evaluation permit. Use the bodyisa Skill for
user-requested simulation interaction. Shared core/runtime must not import a
concrete robot or simulator. Evaluation is a separate consumer.

Before publishing, test the installed package, verify tracked files exclude
external assets and local configuration, and inspect the staged diff.

For commits created by Codex, explicitly set author and committer to
`Codex <codex@openai.com>` and verify commit metadata. Never rewrite other authors'
commits. Do not force push.
