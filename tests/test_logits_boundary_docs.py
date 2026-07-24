from pathlib import Path


AB_DOC = Path("docs/logits-boundary-ab.md")
PUBLIC_DOCS = [
    AB_DOC,
    Path("docs/where-optimizations-stop-mattering.md"),
    Path("docs/experiment-status.md"),
    Path("docs/logits-boundary-rfc.md"),
    Path("README.md"),
    Path("benchmarks/results/README.md"),
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_logits_boundary_ab_links_core_evidence():
    text = read(AB_DOC)
    assert "[LM-head / logits / sampling boundary](logits-boundary-rfc.md)" in text
    assert (
        "../benchmarks/results/l20-vllm-logits-boundary-rfc-shadow/"
        "qwen3-0p6b-o2-v1/"
    ) in text
    assert "benchmarks/results/l20-vllm-logits-boundary-trace/" in text


def test_public_summaries_link_the_ab_plan_once():
    readme_link = (
        "| Logits-boundary A/B plan | "
        "[`docs/logits-boundary-ab.md`](docs/logits-boundary-ab.md) |"
    )
    summary_link = "[`docs/logits-boundary-ab.md`](logits-boundary-ab.md)"
    assert read(Path("README.md")).count(readme_link) == 1
    assert read(Path("docs/where-optimizations-stop-mattering.md")).count(
        summary_link
    ) == 1


def test_logits_boundary_docs_track_gemm_epilogue_scout():
    rfc = read(Path("docs/logits-boundary-rfc.md"))
    status = read(Path("docs/experiment-status.md"))
    results_index = read(Path("benchmarks/results/README.md"))
    artifact = "benchmarks/results/l20-vllm-gemm-epilogue-scout/"
    assert artifact in rfc
    assert artifact in status
    assert "l20-vllm-gemm-epilogue-scout/" in results_index
    assert "f1cf6b0-clean-upstream" in rfc
    assert "f1cf6b0-clean-upstream" in status
    assert "clean upstream" in results_index
    assert "l20-vllm-gemm-epilogue-trace/" in results_index
    assert "f1cf6b0-clean-install-smoke" in rfc
    assert "install_l20_gemm_epilogue_trace.py" in rfc
    assert "install_l20_gemm_epilogue_trace.py" in read(Path("integrations/vllm/README.md"))
    assert "LogitsProcessor" in rfc
    assert "ParallelLMHead" in rfc
    assert "TopKTopPSampler" in rfc
    assert "sampler-only hook is too late" in rfc
    assert "not a sampler-only hook" in status


def test_ab_doc_keeps_shadow_and_performance_claims_separate():
    text = read(AB_DOC)
    assert "What The Shadow Trace Proves" in text
    assert "What Is Not Proven Yet" in text
    assert "does not prove that an epilogue improves ITL" in text
    assert "not a latency improvement" in text
    assert "paired serving JSON" in text
    assert "not as a proven serving win" in text


def test_public_docs_avoid_unsupported_epilogue_itl_claims():
    unsupported_phrases = [
        "epilogue already improves ITL",
        "epilogue improves ITL today",
        "epilogue has improved ITL",
        "proven ITL improvement",
        "confirmed ITL win",
        "is a proven serving win",
        "as a proven serving win today",
        "shadow hook speedup",
        "shadow trace speedup",
    ]
    combined = "\n".join(read(path) for path in PUBLIC_DOCS).lower()
    for phrase in unsupported_phrases:
        assert phrase.lower() not in combined
