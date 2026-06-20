import io
import json
import unittest
from contextlib import redirect_stdout

from l20_stack.cli import main


class CliTest(unittest.TestCase):
    def test_plan_outputs_json(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["plan", "--config", "configs/qlora_l20.json"])

        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["task"], "qlora-smoke-plan")
        self.assertIn("total_gib", payload["estimate"])
        self.assertTrue(payload["estimate"]["fits_device"])

    def test_operator_plan_outputs_json(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["operator-plan", "--config", "configs/l20_operator_targets.json"])

        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["summary"]["compile_target"], "sm_89")
        self.assertEqual(payload["plans"][0]["name"], "rmsnorm")
        self.assertEqual(payload["plans"][0]["roofline_class"], "memory_bound")


if __name__ == "__main__":
    unittest.main()
