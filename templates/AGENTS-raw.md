# AGENTS.md

Reverse engineering of <TARGET> (<what the image is, where it came from>), on the `<platform>`
platform. Read this file first.

## Start here

1. `./dk status`: current state and the next command to run.
2. [docs/overview.md](docs/overview.md): what is where in this image, and what is still closed.
3. [playbook/README.md](playbook/README.md): portable rules; start with `playbook/platforms/raw.md`.

## Task → command

| Task | Command |
|---|---|
| Where things stand / what next | `./dk status` |
| Map the image: gaps, regions, entry points, containers, files, strings | `./dk map [--limit N]` |
| Carve regions, containers and files, or verify them against the image | `./dk carve [--check]` |
| Disassemble (CPU and link address from the region's vector table, else `--arch`/`--va`) | `./dk dis --at OFF [--va VA] [--length N] [--arch A]` |
| Toolkit tests / docs check | `./dk test` / `./dk doc-lint` |
| Sync the shared toolkit repo with this copy | `./dk toolkit diff\|export\|import` |

Exit codes: `0` ok, `1` mismatch, `2` usage/environment, `3` build error. Every command ends with a `next:` hint.

## Rules

- **Never commit the image or anything carved from it.** The image and `build/` are git-ignored.
- **Facts in `docs/` are verified against the bytes.** Unverified ideas are marked as hypotheses.
- **Toolkit changes:** make them here, `./dk test`, commit, `./dk toolkit export`, then commit in the toolkit repo.
