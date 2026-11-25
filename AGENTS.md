# Project instructions

## Response style

- Load the `caveman` skill with the `skill` tool at the start of every session in
  this project, before answering the first prompt.
- Follow the `caveman` skill for every response in that session, for every agent
  used in this folder.
- If the skill is no longer in context (for example after compaction), load it
  again before the next answer.
- Switch back to normal prose only when the user says "stop caveman" or
  "normal mode". A new session starts in caveman mode again.

## Memory

- Read `TODO.md` and `memory/MEMORY.md` at the start of every session.
- Treat the newest entry in `memory/MEMORY.md` as the current state of the work.
- Before ending a session, update `TODO.md` and append a new dated entry at the
  top of `memory/MEMORY.md` with: goal, decisions, files changed, blockers, next
  step.
- Append entries; never delete or rewrite older ones except to fix a factual
  error.
- Write memory and TODO entries in normal prose, not caveman voice.
