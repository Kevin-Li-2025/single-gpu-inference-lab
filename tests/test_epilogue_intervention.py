import importlib.util
import json

import pytest

from l20_stack.epilogue.intervention import (
    CONTINUE_EPILOGUE_PROTOTYPE,
    DO_NOT_CLAIM_WIN,
    NEEDS_MORE_RUNS,
    render_logits_boundary_ab_markdown,
    summarize_logits_boundary_ab,
)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_report(run_dir, concurrency, input_tokens, run, median_itl_ms, output_throughput):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed": 4,
        "failed": 0,
        "request_throughput": output_throughput / 32.0,
        "output_throughput": output_throughput,
        "median_ttft_ms": 20.0,
        "p95_ttft_ms": 30.0,
        "median_itl_ms": median_itl_ms,
        "p95_itl_ms": median_itl_ms + 1.0,
    }
    (run_dir / f"c{concurrency}-i{input_tokens}-r{run}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def write_trace_summary(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logits-boundary-summary.json").write_text(
        json.dumps(
            {
                "total_events": 100,
                "eligible_events": 90,
                "fallback_events": 10,
                "eligible_fraction": 0.9,
                "eligible_logits_mib": 256.0,
                "total_logits_mib": 300.0,
                "shadow_events": 100,
                "shadow_eligible_events": 88,
                "shadow_fallback_events": 12,
                "shadow_eligible_fraction": 0.88,
                "shadow_avoidable_logits_mib": 250.0,
            }
        ),
        encoding="utf-8",
    )


def write_campaign_summary(run_dir, median_itl_ms, output_throughput):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "campaign-summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "serving_report_count": 2,
                "shapes": [
                    {
                        "max_concurrency": 1,
                        "input_tokens": 512,
                        "runs": 2,
                        "metrics": {
                            "median_itl_ms": median_itl_ms,
                            "output_throughput": output_throughput,
                            "request_throughput": output_throughput / 32.0,
                        },
                    }
                ],
                "trace_summary": {
                    "total_events": 10,
                    "eligible_events": 8,
                    "fallback_events": 2,
                    "eligible_fraction": 0.8,
                },
            }
        ),
        encoding="utf-8",
    )


def test_logits_boundary_ab_strict_win_with_trace_and_shadow(tmp_path):
    root = tmp_path / "ab"
    baseline = root / "baseline-trace"
    candidate = root / "candidate-serving"
    write_trace_summary(baseline)
    write_report(baseline, 1, 512, 1, 10.0, 100.0)
    write_report(baseline, 1, 512, 2, 14.0, 120.0)
    write_report(candidate, 1, 512, 1, 9.0, 130.0)
    write_report(candidate, 1, 512, 2, 11.0, 134.0)

    summary = summarize_logits_boundary_ab(root)

    assert summary["status"] == "complete"
    assert summary["verdict"] == CONTINUE_EPILOGUE_PROTOTYPE
    assert summary["incomplete"] is False
    assert summary["gate"]["strict_win_shapes"] == 1
    assert summary["baseline"]["serving_report_count"] == 2
    assert summary["candidate"]["serving_report_count"] == 2
    assert summary["baseline"]["trace_eligibility"]["present"] is True
    assert summary["baseline"]["trace_eligibility"]["shadow_present"] is True
    assert summary["candidate"]["trace_eligibility"]["present"] is False

    shape = summary["shapes"][0]
    assert shape["shape"] == "c1-i512"
    assert shape["baseline"]["median_itl_ms"] == 12.0
    assert shape["candidate"]["median_itl_ms"] == 10.0
    assert shape["baseline"]["output_throughput"] == 110.0
    assert shape["candidate"]["output_throughput"] == 132.0
    assert shape["deltas"]["median_itl_ms_pct"] == pytest.approx(-16.666666)
    assert shape["deltas"]["output_throughput_pct"] == pytest.approx(20.0)
    assert shape["strict_win"] is True
    assert shape["incomplete"] is False

    markdown = render_logits_boundary_ab_markdown(summary)
    assert "continue_epilogue_prototype" in markdown
    assert "c1-i512" in markdown
    assert "90 / 100" in markdown


def test_logits_boundary_ab_reports_incomplete_when_candidate_dir_missing(tmp_path):
    root = tmp_path / "ab"
    baseline = root / "baseline-trace"
    write_report(baseline, 1, 512, 1, 10.0, 100.0)

    summary = summarize_logits_boundary_ab(root)

    assert summary["status"] == "incomplete"
    assert summary["verdict"] == NEEDS_MORE_RUNS
    assert summary["incomplete"] is True
    assert "missing_candidate_dir" in summary["incomplete_reasons"]
    assert "c1-i512:missing_candidate_shape" in summary["incomplete_reasons"]


def test_logits_boundary_ab_does_not_claim_complete_non_strict_win(tmp_path):
    root = tmp_path / "ab"
    baseline = root / "baseline-trace"
    candidate = root / "candidate-serving"
    write_report(baseline, 1, 512, 1, 10.0, 100.0)
    write_report(baseline, 1, 512, 2, 10.0, 100.0)
    write_report(candidate, 1, 512, 1, 9.0, 95.0)
    write_report(candidate, 1, 512, 2, 9.0, 95.0)

    summary = summarize_logits_boundary_ab(root)

    assert summary["status"] == "complete"
    assert summary["verdict"] == DO_NOT_CLAIM_WIN
    assert summary["shapes"][0]["median_itl_win"] is True
    assert summary["shapes"][0]["throughput_win"] is False
    assert summary["shapes"][0]["strict_win"] is False


def test_logits_boundary_ab_uses_campaign_summary_and_cli_writes_outputs(tmp_path):
    root = tmp_path / "ab"
    write_campaign_summary(root / "baseline-trace", 12.0, 100.0)
    write_campaign_summary(root / "candidate-serving", 10.0, 125.0)
    script = load_module(
        "scripts/summarize_l20_logits_boundary_ab.py",
        "summarize_l20_logits_boundary_ab",
    )
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"

    assert script.main(
        [
            str(root),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    ) == 0

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["verdict"] == CONTINUE_EPILOGUE_PROTOTYPE
    assert payload["baseline"]["shapes"][0]["source"] == "campaign-summary.json"
    assert "L20 Logits Boundary A/B Verdict" in output_md.read_text(encoding="utf-8")


def test_logits_boundary_ab_missing_input_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        summarize_logits_boundary_ab(tmp_path / "missing")
