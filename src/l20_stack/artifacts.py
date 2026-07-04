"""Artifact index checks for checked-in benchmark evidence."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


INDEX_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|")
COMPACT_ARTIFACT_NAMES = frozenset(
    {
        "README.md",
        "summary.json",
        "campaign-summary.json",
        "run-config.json",
    }
)


@dataclass(frozen=True)
class ArtifactIndexEntry:
    reference: str
    path: str
    exists: bool
    has_direct_artifact: bool
    has_nested_artifact: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactIndexReport:
    index_path: str
    result_root: str
    entries: tuple[ArtifactIndexEntry, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "index_path": self.index_path,
            "result_root": self.result_root,
            "ok": self.ok,
            "entry_count": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def inspect_artifact_index(
    index_path: str | Path = "benchmarks/results/README.md",
    *,
    result_root: str | Path = "benchmarks/results",
) -> ArtifactIndexReport:
    index = Path(index_path)
    root = Path(result_root)
    references = parse_index_references(index)
    entries = tuple(_inspect_reference(reference, root) for reference in references)
    errors = []
    warnings = []
    seen: set[str] = set()
    for entry in entries:
        if entry.reference in seen:
            warnings.append(f"duplicate artifact reference: {entry.reference}")
        seen.add(entry.reference)
        if not entry.exists:
            errors.append(f"missing artifact directory: {entry.reference}")
        elif not entry.has_direct_artifact and not entry.has_nested_artifact:
            warnings.append(f"artifact directory has no compact evidence file: {entry.reference}")
    if not entries:
        errors.append(f"no artifact references found in {index}")

    return ArtifactIndexReport(
        index_path=str(index),
        result_root=str(root),
        entries=entries,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def parse_index_references(index_path: str | Path) -> tuple[str, ...]:
    index = Path(index_path)
    references = []
    for line in index.read_text(encoding="utf-8").splitlines():
        match = INDEX_ROW_RE.match(line.strip())
        if match:
            references.append(match.group(1))
    return tuple(references)


def _inspect_reference(reference: str, result_root: Path) -> ArtifactIndexEntry:
    path = _resolve_reference(reference, result_root)
    return ArtifactIndexEntry(
        reference=reference,
        path=str(path),
        exists=path.exists(),
        has_direct_artifact=_has_direct_artifact(path),
        has_nested_artifact=_has_nested_artifact(path),
    )


def _resolve_reference(reference: str, result_root: Path) -> Path:
    normalized = reference.strip().rstrip("/")
    if normalized.startswith(str(result_root) + "/"):
        return Path(normalized)
    return result_root / normalized


def _has_direct_artifact(path: Path) -> bool:
    if not path.exists():
        return False
    return any((path / artifact).exists() for artifact in COMPACT_ARTIFACT_NAMES)


def _has_nested_artifact(path: Path) -> bool:
    if not path.exists():
        return False
    return any(
        child.is_file() and child.name in COMPACT_ARTIFACT_NAMES
        for child in _iter_nested_files(path)
    )


def _iter_nested_files(path: Path) -> Iterable[Path]:
    for child in path.rglob("*"):
        if child.is_file():
            yield child
