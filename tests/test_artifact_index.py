import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from l20_stack.artifacts import inspect_artifact_index, parse_index_references
from l20_stack.cli import main


class ArtifactIndexTest(unittest.TestCase):
    def test_parse_index_references_from_markdown_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = Path(tmpdir) / "README.md"
            index.write_text(
                "| Result directory | Status | Why it matters |\n"
                "| --- | --- | --- |\n"
                "| `a100-example/` | Confirmed | evidence |\n"
                "| not a reference | ignored | ignored |\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_index_references(index), ("a100-example/",))

    def test_missing_artifact_directory_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "benchmarks" / "results"
            root.mkdir(parents=True)
            index = root / "README.md"
            index.write_text(
                "| Result directory | Status | Why it matters |\n"
                "| --- | --- | --- |\n"
                "| `missing/` | Smoke | no artifact yet |\n",
                encoding="utf-8",
            )
            report = inspect_artifact_index(index, result_root=root)
            self.assertFalse(report.ok)
            self.assertIn("missing artifact directory: missing/", report.errors)

    def test_current_repo_artifact_index_has_no_missing_references(self):
        report = inspect_artifact_index()
        self.assertTrue(report.ok, report.errors)
        self.assertGreater(report.to_dict()["entry_count"], 20)

    def test_cli_artifact_index_emits_json_report(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["artifact-index"])
        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["ok"])
        self.assertGreater(payload["entry_count"], 20)


if __name__ == "__main__":
    unittest.main()
