# Brief: {{role}}, slice {{slice}}

{{scope}}

## Rules (every agent)

- Work only in your own worktree: `./dk worktree add {{slice}}`, then run every command from `worktrees/{{slice}}`. A half-done edit there cannot break anyone else's build.
- Commit only through the tool: `./dk save <unit…> --note "what's left"` for progress, `./dk accept <unit>` once it is exact, `./dk commit-tools PATH… -m MSG` for anything else. Each syncs your worktree with the shared branch, proves the result in a clean tree and lands it. Never run raw git `commit`, `reset` or `stash`.
- Exact matches only, through the review checklist in `playbook/agents.md`. No new `volatile`, pragma devices, dummy members or forbidden patterns (`./dk quality`).
- Iterate with `./dk check <unit> --private`. After an edit to a header, a linked unit or a symbol name, run the full link (`./dk build`).
- Headers other agents include: re-read just before editing, keep the edit additive, land it promptly. A cross-unit change (a rename of a linked symbol, a header retype and the units that need it) lands as one `./dk commit-tools PATH… -m MSG` from your worktree, after `./dk check --all` there reports MATCH. Its owner (e.g. a header family's owner) decides the content; the coordinator can't see your worktree's edits.
- Pick up what others landed with `./dk worktree sync`. It refuses on a conflict: fix it by hand, or `--keep-mine PATH`.
- Keep going until told to wind down. Then finish the current function, land it, write your notes (below) and report what landed and what is open.
{{#sweeper}}

## Your job: sweep the slice

Progress counts code bytes: get the most bytes exact. Work down the units with the most open code, big functions you can get exact first. Leave register-only near misses to a closer.

{{undeclared}}

{{units}}

Nearly exact in this slice (cheap wins, or leave them to a closer):

{{near}}
{{/sweeper}}
{{#closer}}

## Your job: take nearly-exact units to linked

Cheapest first: fix what blocks each unit, then `./dk accept <unit>`. `./dk diff <unit> --function F` shows what differs in a function; the playbook's matching notes list the verified rewrites.

{{near}}

Everything still open in scope:

{{units}}
{{/closer}}

## Where to look

{{playbook}}

## Your notes from earlier sessions

{{notes}}

Before you stop, or when your context runs low, write what you learned and what is still open to `{{notes_path}}`. The next agent on this slice gets it in its brief.
