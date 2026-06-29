import importlib.util
import json


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flashsampling_trace_summary_counts_reasons_and_bytes(tmp_path):
    module = load_module(
        "scripts/summarize_l20_flashsampling_trace.py",
        "summarize_l20_flashsampling_trace",
    )
    events = [
        {
            "eligible": True,
            "metadata": {
                "flashsampling_epilogue": {
                    "would_use_epilogue": True,
                    "fallback_reasons": [],
                    "logits_materialization_bytes": 20,
                    "avoidable_logits_materialization_bytes": 20,
                    "flashsampling_request": {
                        "batch_size": 1,
                        "hidden_size": 512,
                        "vocab_size": 10,
                        "sampling_mode": "gumbel",
                    },
                }
            },
        },
        {
            "eligible": False,
            "metadata": {
                "flashsampling_epilogue": {
                    "would_use_epilogue": False,
                    "fallback_reasons": ["top_k_top_p_unsupported", "penalties"],
                    "logits_materialization_bytes": 40,
                    "avoidable_logits_materialization_bytes": 0,
                    "flashsampling_request": {
                        "batch_size": 2,
                        "hidden_size": 512,
                        "vocab_size": 10,
                        "sampling_mode": "gumbel",
                    },
                }
            },
        },
    ]
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    summary = module.summarize(module.read_events(trace))

    assert summary["total_events"] == 2
    assert summary["eligible_events"] == 1
    assert summary["eligible_fraction"] == 0.5
    assert summary["avoidable_logits_bytes"] == 20
    assert summary["total_logits_bytes"] == 60
    assert summary["reason_counts"] == {"penalties": 1, "top_k_top_p_unsupported": 1}
    assert summary["shape_counts"] == {
        "b1-h512-v10-gumbel": 1,
        "b2-h512-v10-gumbel": 1,
    }
