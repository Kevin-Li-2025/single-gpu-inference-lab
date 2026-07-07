"""Local Markdown path checks for docs and README files."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
ALLOWED_LOCAL_PREFIXES = (
    ".github/",
    "benchmarks/",
    "configs/",
    "cpp/",
    "cuda/",
    "docs/",
    "integrations/",
    "scripts/",
    "src/",
    "tests/",
)
ALLOWED_LOCAL_FILES = {"README.md", "LICENSE", "pyproject.toml"}


@dataclass(frozen=True)
class DocLinkEntry:
    source: str
    line: int
    reference: str
    path: str
    exists: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DocLinkReport:
    root: str
    checked_files: tuple[str, ...]
    entries: tuple[DocLinkEntry, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "ok": self.ok,
            "checked_files": list(self.checked_files),
            "entry_count": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
            "errors": list(self.errors),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def inspect_doc_links(
    root: str | Path = ".",
    *,
    files: Iterable[str | Path] | None = None,
) -> DocLinkReport:
    base = Path(root)
    markdown_files = tuple(_default_markdown_files(base) if files is None else files)
    entries: list[DocLinkEntry] = []
    errors: list[str] = []
    checked: list[str] = []
    for markdown in markdown_files:
        source = Path(markdown)
        if not source.is_absolute():
            source = base / source
        if not source.exists():
            errors.append(f"missing markdown file: {_display_path(source, base)}")
            continue
        checked.append(_display_path(source, base))
        for line_no, line in enumerate(_strip_fenced_blocks(source), 1):
            for reference in _line_references(line):
                resolved = _resolve_reference(reference, source.parent, base)
                if resolved is None:
                    continue
                exists = resolved.exists()
                entry = DocLinkEntry(
                    source=_display_path(source, base),
                    line=line_no,
                    reference=reference,
                    path=_display_path(resolved, base),
                    exists=exists,
                )
                entries.append(entry)
                if not exists:
                    errors.append(
                        f"{entry.source}:{line_no}: missing local path {reference}"
                    )
    return DocLinkReport(
        root=str(base),
        checked_files=tuple(checked),
        entries=tuple(entries),
        errors=tuple(errors),
    )


def _default_markdown_files(root: Path) -> tuple[Path, ...]:
    candidates = [
        root / "README.md",
        root / "benchmarks" / "results" / "README.md",
        root / "docs" / "experiment-status.md",
        root / "docs" / "hardware-scope.md",
        root / "docs" / "repo-map.md",
        root / "docs" / "where-optimizations-stop-mattering.md",
    ]
    return tuple(path for path in candidates if path.exists())


def _strip_fenced_blocks(path: Path) -> list[str]:
    lines: list[str] = []
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            lines.append("")
            continue
        lines.append("" if in_fence else line)
    return lines


def _line_references(line: str) -> tuple[str, ...]:
    references: list[str] = []
    references.extend(match.group(1).split("#", 1)[0] for match in MARKDOWN_LINK_RE.finditer(line))
    for match in CODE_SPAN_RE.finditer(line):
        text = match.group(1).strip()
        if _looks_like_local_reference(text):
            references.append(text)
    return tuple(reference for reference in references if reference)


def _looks_like_local_reference(reference: str) -> bool:
    if "\n" in reference or " " in reference or "\\" in reference:
        return False
    if any(char in reference for char in "*?{}[]"):
        return False
    stripped = reference.strip().strip(",.;:")
    if stripped in ALLOWED_LOCAL_FILES:
        return True
    return stripped.startswith(ALLOWED_LOCAL_PREFIXES)


def _resolve_reference(reference: str, source_dir: Path, root: Path) -> Path | None:
    normalized = reference.strip().strip(",.;:")
    if not normalized or normalized.startswith(("#", "http://", "https://", "mailto:")):
        return None
    if not _looks_like_local_reference(normalized):
        return None
    if normalized.startswith("./") or normalized.startswith("../"):
        return (source_dir / normalized).resolve()
    return root / normalized.rstrip("/")


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
