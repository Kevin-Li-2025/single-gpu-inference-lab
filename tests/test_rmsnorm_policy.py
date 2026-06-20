import unittest

from l20_stack.ops.rmsnorm_policy import build_residual_rmsnorm_policy


def shape_report(rows, hidden_size, eager_ms, l20_ms, flashinfer_ms):
    return {
        "shape": {"rows": rows, "hidden_size": hidden_size, "dtype": "float16"},
        "operators": {
            "residual_rmsnorm": {
                "providers": {
                    "torch_eager": {
                        "correct": True,
                        "timing_ms": {"p50": eager_ms},
                    },
                    "l20_inplace": {
                        "correct": True,
                        "timing_ms": {"p50": l20_ms},
                    },
                    "flashinfer": {
                        "correct": True,
                        "timing_ms": {"p50": flashinfer_ms},
                    },
                }
            }
        },
    }


class RmsNormPolicyTest(unittest.TestCase):
    def test_policy_prefers_stable_measured_backend(self):
        reports = [
            {"shapes": [shape_report(32, 4096, 0.013, 0.007, 0.009)]},
            {"shapes": [shape_report(32, 4096, 0.014, 0.007, 0.009)]},
            {"shapes": [shape_report(32, 4096, 0.013, 0.008, 0.009)]},
        ]

        policy = build_residual_rmsnorm_policy(reports)[0]

        self.assertEqual(policy.fastest_provider, "l20_inplace")
        self.assertEqual(policy.recommended_backend, "triton")
        self.assertTrue(policy.stable)
        self.assertEqual(policy.source_runs, 3)

    def test_policy_marks_small_margin_as_unstable(self):
        reports = [
            {"shapes": [shape_report(512, 5120, 0.048, 0.0317, 0.0320)]},
            {"shapes": [shape_report(512, 5120, 0.048, 0.0318, 0.0321)]},
            {"shapes": [shape_report(512, 5120, 0.048, 0.0319, 0.0322)]},
        ]

        policy = build_residual_rmsnorm_policy(reports)[0]

        self.assertEqual(policy.fastest_provider, "l20_inplace")
        self.assertFalse(policy.stable)


if __name__ == "__main__":
    unittest.main()
