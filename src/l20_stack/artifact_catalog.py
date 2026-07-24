"""Build a compact machine-readable catalog from benchmark result artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from l20_stack.artifacts import parse_index_references


TABLE_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*(.*?)\s*\|$")


@dataclass(frozen=True)
class ArtifactCatalogEntry:
    reference: str
    path: str
    status: str
    category: str
    summary: str
    has_readme: bool
    has_summary_json: bool
    has_campaign_summary_json: bool
    has_evidence_status_json: bool
    compact_file_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactCatalog:
    index_path: str
    result_root: str
    entries: tuple[ArtifactCatalogEntry, ...]
    category_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "index_path": self.index_path,
            "result_root": self.result_root,
            "entry_count": len(self.entries),
            "category_counts": self.category_counts,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def build_artifact_catalog(
    index_path: str | Path = "benchmarks/results/README.md",
    *,
    result_root: str | Path = "benchmarks/results",
) -> ArtifactCatalog:
    index = Path(index_path)
    root = Path(result_root)
    rows = _parse_index_rows(index)
    entries = tuple(_build_entry(reference, rows.get(reference, {}), root) for reference in rows)
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.category] = counts.get(entry.category, 0) + 1
    return ArtifactCatalog(
        index_path=str(index),
        result_root=str(root),
        entries=entries,
        category_counts=dict(sorted(counts.items())),
    )


def _parse_index_rows(index: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    known_references = set(parse_index_references(index))
    for line in index.read_text(encoding="utf-8").splitlines():
        match = TABLE_ROW_RE.match(line.strip())
        if not match:
            continue
        reference, status, summary = match.groups()
        if reference not in known_references:
            continue
        rows[reference] = {
            "status": _clean_cell(status),
            "summary": _clean_cell(summary),
        }
    return rows


def _build_entry(
    reference: str,
    row: dict[str, str],
    result_root: Path,
) -> ArtifactCatalogEntry:
    path = _resolve_reference(reference, result_root)
    compact_files = _compact_files(path)
    names = {file.name for file in compact_files}
    status = row.get("status", "")
    return ArtifactCatalogEntry(
        reference=reference,
        path=str(path),
        status=status,
        category=_category(status, row.get("summary", "")),
        summary=row.get("summary", ""),
        has_readme="README.md" in names,
        has_summary_json="summary.json" in names,
        has_campaign_summary_json="campaign-summary.json" in names,
        has_evidence_status_json="evidence-status.json" in names,
        compact_file_count=len(compact_files),
    )


def _resolve_reference(reference: str, result_root: Path) -> Path:
    normalized = reference.strip().rstrip("/")
    if normalized.startswith(str(result_root) + "/"):
        return Path(normalized)
    return result_root / normalized


def _compact_files(path: Path) -> tuple[Path, ...]:
    if not path.exists():
        return ()
    return tuple(
        file
        for file in path.rglob("*")
        if file.is_file()
        and file.name
        in {
            "README.md",
            "summary.json",
            "campaign-summary.json",
            "evidence-status.json",
            "run-config.json",
        }
    )


def _category(status: str, summary: str) -> str:
    status_text = status.lower()
    summary_text = summary.lower()
    if "superseded" in status_text or "superseded" in summary_text:
        return "superseded"
    if "source-map" in status_text or "source map" in status_text:
        return "path_proof"
    if "negative" in status_text:
        return "negative"
    if "active" in status_text or "current p0" in status_text:
        return "active"
    if "smoke" in status_text or "path proof" in status_text or "proof" in status_text:
        return "path_proof"
    if "positive" in status_text or "confirmed" in status_text or "win" in status_text:
        return "positive"
    if "analysis" in status_text or "paper" in status_text:
        return "analysis"
    if "negative" in summary_text or "regresses" in summary_text or "slower" in summary_text:
        return "negative"
    if "proof" in summary_text or "trace" in summary_text:
        return "path_proof"
    if "positive" in summary_text or "wins" in summary_text or "improves" in summary_text:
        return "positive"
    return "other"


def _clean_cell(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("<br>", " ")).strip()
