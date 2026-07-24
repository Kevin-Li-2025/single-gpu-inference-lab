import json
from pathlib import Path


SUPERSEDED_READMES = {
    "benchmarks/results/l20-gpu-sampling/README.md": "Superseded",
    "benchmarks/results/l20-vllm-sampling-itl/README.md": "Superseded",
    (
        "benchmarks/results/l20-logits-boundary-ab-smoke/"
        "qwen25-coder-1p5b-c1c4-i512-r1/README.md"
    ): "Superseded",
    "benchmarks/results/l20-vllm-compiled-sampler-scout/README.md": "Source-map",
    "benchmarks/results/l20-vllm-compiled-sampler-scout-v2/README.md": "Source-map",
    "benchmarks/results/nsys/sampling/README.md": "Path proof only",
    "benchmarks/results/a100-fused-topk-topp-penalty/README.md": "Superseded",
    "benchmarks/results/a100-sparse-topk-topp-penalty/README.md": "Superseded",
    "benchmarks/results/a100-vllm-sparse-penalty-sampling/README.md": "Superseded",
    "benchmarks/results/a100-vllm-combined-sampling-logprobs/README.md": "Superseded",
    "benchmarks/results/a100-vllm-combined-sampling-logprobs-matrix/README.md": (
        "Superseded"
    ),
}


SUPERSEDED_SUMMARIES = (
    "benchmarks/results/l20-vllm-sampling-itl/qwen25-coder-1p5b-summary.json",
    (
        "benchmarks/results/l20-logits-boundary-ab-smoke/"
        "qwen25-coder-1p5b-c1c4-i512-r1/summary.json"
    ),
    "benchmarks/results/a100-fused-topk-topp-penalty/summary.json",
    "benchmarks/results/a100-vllm-sparse-penalty-sampling/summary.json",
    "benchmarks/results/a100-vllm-flashinfer-sparse-penalty-sampling/summary.json",
    "benchmarks/results/a100-vllm-combined-sampling-logprobs-matrix/summary.json",
)


def test_affected_sampling_readmes_show_evidence_status():
    for path, marker in SUPERSEDED_READMES.items():
        text = Path(path).read_text(encoding="utf-8")
        assert marker in text[:700], path


def test_affected_sampling_summaries_are_machine_readable_as_superseded():
    for path in SUPERSEDED_SUMMARIES:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["evidence_status"] == "superseded_semantics", path


def test_all_superseded_summary_files_are_explicitly_non_comparable():
    for path in Path("benchmarks/results").rglob("summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("evidence_status") != "superseded_semantics":
            continue
        assert payload["performance_comparable"] is False, path
        assert "delta" not in payload, path


def test_curated_index_does_not_publish_sampler_ab_as_current_negative():
    index = Path("benchmarks/results/README.md").read_text(encoding="utf-8")
    row = next(
        line
        for line in index.splitlines()
        if "`l20-logits-boundary-ab-smoke/`" in line
    )
    assert "Superseded A/B / path proof" in row
    assert "currently regresses" not in row


def test_logits_boundary_checked_summary_matches_non_comparable_schema():
    path = (
        "benchmarks/results/l20-logits-boundary-ab-smoke/"
        "qwen25-coder-1p5b-c1c4-i512-r1/summary.json"
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["evidence_status"] == "superseded_semantics"
    assert payload["performance_comparable"] is False
    assert payload["collection_status"] == "complete"
    assert payload["status"] == "not_comparable"
    assert payload["verdict"] == "not_comparable"
    assert payload["historical_verdict"] == "do_not_claim_win"


def test_triangle_generators_default_to_unvalidated_semantics():
    for path in (
        "scripts/summarize_vllm_sparse_sampling_ab.py",
        "scripts/summarize_vllm_sparse_penalty_triangle.py",
        "scripts/summarize_vllm_sparse_penalty_triangle_matrix.py",
    ):
        source = Path(path).read_text(encoding="utf-8")
        assert "requires_semantic_validation" in source
        assert "performance_comparable" in source
        assert "not current" in source


def test_combined_top_logprobs_runner_marks_sparse_sampler_results_unvalidated():
    source = Path("scripts/run_vllm_a100_top_logprobs_ab.sh").read_text(
        encoding="utf-8"
    )
    assert 'summary["evidence_status"] = "requires_semantic_validation"' in source
    assert 'summary["performance_comparable"] = False' in source
    assert 'summary["historical_delta"] = summary.pop("delta")' in source
    assert "Historical result (not current evidence)" in source


def test_raw_sampler_directories_have_machine_readable_status_manifests():
    for directory in (
        "benchmarks/results/l20-gpu-sampling",
        "benchmarks/results/a100-fused-topk-topp-penalty",
        "benchmarks/results/a100-sparse-topk-topp-penalty",
    ):
        payload = json.loads(
            (Path(directory) / "evidence-status.json").read_text(encoding="utf-8")
        )
        assert payload["schema_version"] == 1
        assert payload["evidence_status"] == "superseded_semantics"
        assert payload["performance_comparable"] is False
        assert payload["reason"]
        assert payload["revalidation_required"]
        assert payload["scope"]


def test_machine_catalog_includes_all_affected_sampler_directories():
    payload = json.loads(
        Path("benchmarks/results/artifact-catalog.json").read_text(encoding="utf-8")
    )
    entries = {entry["reference"]: entry for entry in payload["entries"]}
    expected = {
        "l20-gpu-sampling/": ("superseded", True),
        "a100-fused-topk-topp-penalty/": ("superseded", True),
        "a100-sparse-topk-topp-penalty/": ("superseded", True),
        "l20-vllm-sampling-itl/": ("superseded", False),
        "l20-vllm-compiled-sampler-scout/": ("path_proof", False),
        "l20-vllm-compiled-sampler-scout-v2/": ("path_proof", False),
    }
    for reference, (category, has_manifest) in expected.items():
        assert entries[reference]["category"] == category
        assert entries[reference]["has_evidence_status_json"] is has_manifest
