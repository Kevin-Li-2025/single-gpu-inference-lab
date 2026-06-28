import importlib.util
import json
from pathlib import Path


def load_ceiling_script():
    path = Path("scripts/analyze_serving_optimization_ceiling.py")
    spec = importlib.util.spec_from_file_location("analyze_serving_optimization_ceiling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_amdahl_speedup_matches_known_case():
    module = load_ceiling_script()
    assert module.amdahl_speedup(50.0, 2.0) == 4 / 3
    assert module.amdahl_speedup(50.0, float("inf")) == 2.0
    assert module.amdahl_speedup(0.0, 2.0) == 1.0


def test_boundary_summary_groups_gemm_and_sampling():
    module = load_ceiling_script()
    summary = {
        "gpu": {
            "families": {
                "cutlass_or_cublas_gemm": {"time_pct": 40.0, "total_time_ns": 40_000_000},
                "cublas_gemv": {"time_pct": 5.0, "total_time_ns": 5_000_000},
                "flashinfer_sampling": {"time_pct": 1.0, "total_time_ns": 1_000_000},
                "sampler_other": {"time_pct": 2.0, "total_time_ns": 2_000_000},
                "pytorch_softmax": {"time_pct": 3.0, "total_time_ns": 3_000_000},
            }
        }
    }
    result = module.boundary_summary(summary, "gpu", module.GPU_BOUNDARIES)
    assert result["gemm_or_gemv"]["time_pct"] == 45.0
    assert result["gemm_or_gemv"]["total_time_ms"] == 45.0
    assert result["standalone_sampling"]["time_pct"] == 6.0


def test_recommendations_stop_low_ceiling_sampling_and_custom_kernel():
    module = load_ceiling_script()
    runs = [
        {
            "gpu_boundaries": {
                "gemm_or_gemv": {"time_pct": 62.0},
                "standalone_sampling": {"time_pct": 3.4},
                "custom_l20_current": {"time_pct": 1.6},
                "metadata_fill": {"time_pct": 14.0},
            },
            "api_boundaries": {
                "launch_sync_transfer": {"time_pct": 52.0},
            },
        }
    ]
    lm_head = {
        "best_candidate": {
            "ratio": 1.02,
            "ratio_name": "triton_top1_over_full_logits_top1",
        }
    }
    recommendations = module.build_recommendations(runs, lm_head)
    priorities = {(row["priority"], row["target"]) for row in recommendations}
    assert ("P0", "production GEMM/GEMV epilogue or upstream logits boundary") in priorities
    assert ("P0", "avoid standalone LM-head replacement") in priorities
    assert ("Stop", "standalone sampling kernels") in priorities
    assert (
        "Stop",
        "micro-optimizing the existing Q/K/RoPE/KV kernel alone",
    ) in priorities


def test_lm_head_summary_reads_existing_result(tmp_path):
    module = load_ceiling_script()
    result_path = tmp_path / "lm.json"
    result_path.write_text(
        json.dumps(
            {
                "shape": {"batch": 1, "hidden": 1536, "vocab": 151936},
                "ratios": {
                    "triton_top1_over_full_logits_top1": 1.022,
                    "full_logits_top1_over_triton_top1": 0.978,
                },
            }
        ),
        encoding="utf-8",
    )
    summary = module.summarize_lm_head([result_path])
    assert summary["candidate_count"] == 1
    assert summary["best_candidate"]["ratio"] == 1.022
