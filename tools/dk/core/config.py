"""Load dk.toml, the only project-specific input to the toolkit."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CONFIG_NAME = "dk.toml"


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class DocsConfig:
    agents_md: Path
    playbook: Path
    project: Path
    generated: Path
    exclude: tuple[Path, ...]
    banned_in_playbook: tuple[str, ...]
    playbook_file_lines: int
    playbook_total_lines: int
    agents_md_lines: int


@dataclass(frozen=True)
class Config:
    root: Path
    name: str
    game_id: str
    platform: str
    original: Path
    built: Path
    sha1: str
    disc_search: tuple[Path, ...]
    configure: tuple[str, ...]
    report: Path
    platform_options: dict[str, Any]
    docs: DocsConfig
    quality: dict[str, Any]
    toolkit_repo: Path | None = None       # the shared toolkit repo ([toolkit] repo)

    def rel(self, path: Path) -> str:
        """Path relative to the repo root, for messages."""
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)


def _get(table: dict[str, Any], key: str, where: str) -> Any:
    if key not in table:
        raise ConfigError(f"{CONFIG_NAME}: missing [{where}] {key}")
    return table[key]


def load(root: Path = REPO) -> Config:
    path = root / CONFIG_NAME
    if not path.is_file():
        raise ConfigError(f"{path} not found")
    with path.open("rb") as f:
        raw = tomllib.load(f)

    project = raw.get("project", {})
    target = raw.get("target", {})
    build = raw.get("build", {})
    docs = raw.get("docs", {})
    budget = docs.get("budget", {})

    def p(value: str) -> Path:
        return (root / value).resolve()

    return Config(
        root=root,
        name=_get(project, "name", "project"),
        game_id=_get(project, "game_id", "project"),
        platform=_get(project, "platform", "project"),
        original=p(_get(target, "original", "target")),
        built=p(_get(target, "built", "target")),
        sha1=str(_get(target, "sha1", "target")).lower(),
        disc_search=tuple(p(d) for d in target.get("disc_search", [])),
        configure=tuple(build.get("configure", ["configure.py"])),
        report=p(build.get("report", "build/report.json")),
        platform_options=dict(raw.get("platform", {})),
        quality=dict(raw.get("quality", {})),
        toolkit_repo=p(raw["toolkit"]["repo"]) if raw.get("toolkit", {}).get("repo") else None,
        docs=DocsConfig(
            agents_md=p(docs.get("agents_md", "AGENTS.md")),
            playbook=p(docs.get("playbook", "playbook")),
            project=p(docs.get("project", "docs/project")),
            generated=p(docs.get("generated", "docs/generated")),
            exclude=tuple(p(e) for e in docs.get("exclude", [])),
            banned_in_playbook=tuple(docs.get("banned_in_playbook", [])),
            playbook_file_lines=int(budget.get("playbook_file_lines", 300)),
            playbook_total_lines=int(budget.get("playbook_total_lines", 1500)),
            agents_md_lines=int(budget.get("agents_md_lines", 100)),
        ),
    )
