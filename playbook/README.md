# Decomp playbook

Portable lessons for matching decompilation and binary reverse engineering.
Nothing here may name a specific project (`./dk doc-lint` enforces this), so the folder can be
copied into the next project unchanged.

| File | Read when |
|---|---|
| [process.md](process.md) | starting a project, or closing a phase |
| [tooling.md](tooling.md) | choosing, pinning or updating tools |
| [agents.md](agents.md) | before running or reviewing agent matching work |
| [matching.md](matching.md) | a function is close but not exact |
| [platforms/gamecube-mwcc.md](platforms/gamecube-mwcc.md) | target is GameCube/Wii built with CodeWarrior |

## Writing entries

- Format: **Rule.** Why. How to detect or apply. No narrative, no history.
- Add only what was verified on a real project and would save time next time.
- Unverified ideas go in the project's docs as hypotheses until something confirms them.
- True for one project only? It belongs in that project's docs, not here.
- When an entry turns out wrong or stale, delete it. Smaller is better.
