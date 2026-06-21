import unittest

from l20_stack.qlora import (
    QLoRAConfig,
    assert_disjoint,
    contamination_report,
    dataset_fingerprint,
    read_jsonl,
)


class QLoRAConfigTest(unittest.TestCase):
    def test_smoke_config_is_valid_and_stable(self):
        config = QLoRAConfig.from_file("configs/qlora_l20_smoke.json")
        self.assertEqual(config.lora_rank, 8)
        self.assertEqual(len(config.fingerprint()), 16)
        large = QLoRAConfig.from_file("configs/qlora_l20_14b_smoke.json")
        self.assertIn("14B", large.model_name)

    def test_fixture_is_disjoint(self):
        train = read_jsonl("tests/fixtures/qlora_train.jsonl")
        evaluation = read_jsonl("tests/fixtures/qlora_eval.jsonl")
        assert_disjoint(train, evaluation)
        self.assertNotEqual(dataset_fingerprint(train), dataset_fingerprint(evaluation))

    def test_overlap_is_rejected(self):
        records = read_jsonl("tests/fixtures/qlora_eval.jsonl")
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint(records, records)

    def test_prompt_contamination_is_reported(self):
        records = read_jsonl("tests/fixtures/qlora_eval.jsonl")
        report = contamination_report(records, records)
        self.assertEqual(report["normalized_prompt_overlap"], 1)


if __name__ == "__main__":
    unittest.main()
