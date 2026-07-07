import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from l20_stack.artifact_catalog import build_artifact_catalog
from l20_stack.cli import main
from l20_stack.doc_links import inspect_doc_links


class DocLinksAndCatalogTest(unittest.TestCase):
    def test_doc_links_validate_allowed_local_references(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scripts").mkdir()
            (root / "scripts" / "run.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            readme = root / "README.md"
            readme.write_text(
                "See `scripts/run.sh` and [missing](docs/missing.md).\n"
                "Ignore model id `Qwen/Qwen2.5-0.5B-Instruct`.\n",
                encoding="utf-8",
            )

            report = inspect_doc_links(root, files=["README.md"])

        self.assertFalse(report.ok)
        self.assertEqual(len(report.entries), 2)
        self.assertIn("missing local path docs/missing.md", report.errors[0])

    def test_current_repo_doc_links_are_valid(self):
        report = inspect_doc_links()
        self.assertTrue(report.ok, report.errors)
        self.assertGreater(report.to_dict()["entry_count"], 20)

    def test_artifact_catalog_parses_status_categories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "benchmarks" / "results"
            result = root / "example-positive"
            result.mkdir(parents=True)
            (result / "README.md").write_text("# result\n", encoding="utf-8")
            zero_regression = root / "example-zero-regressions"
            zero_regression.mkdir()
            (zero_regression / "README.md").write_text("# result\n", encoding="utf-8")
            positive_summary = root / "example-positive-summary"
            positive_summary.mkdir()
            (positive_summary / "README.md").write_text("# result\n", encoding="utf-8")
            index = root / "README.md"
            index.write_text(
                "| Result directory | Status | Why it matters |\n"
                "| --- | --- | --- |\n"
                "| `example-positive/` | Positive serving result | wins on workload |\n"
                "| `example-zero-regressions/` | Boundary result | measured policy has zero regressions |\n"
                "| `example-positive-summary/` | Serving matrix | positive in 4/4 rows |\n",
                encoding="utf-8",
            )

            catalog = build_artifact_catalog(index, result_root=root)

        payload = catalog.to_dict()
        by_reference = {entry["reference"]: entry for entry in payload["entries"]}
        self.assertEqual(payload["entry_count"], 3)
        self.assertEqual(by_reference["example-positive/"]["category"], "positive")
        self.assertEqual(by_reference["example-zero-regressions/"]["category"], "other")
        self.assertEqual(by_reference["example-positive-summary/"]["category"], "positive")
        self.assertEqual(payload["category_counts"]["positive"], 2)

    def test_cli_doc_links_and_artifact_catalog_emit_json(self):
        out = StringIO()
        with redirect_stdout(out):
            doc_code = main(["doc-links", "--file", "README.md"])
        doc_payload = json.loads(out.getvalue())
        self.assertEqual(doc_code, 0)
        self.assertTrue(doc_payload["ok"])

        out = StringIO()
        with redirect_stdout(out):
            catalog_code = main(["artifact-catalog"])
        catalog_payload = json.loads(out.getvalue())
        self.assertEqual(catalog_code, 0)
        self.assertGreater(catalog_payload["entry_count"], 20)


if __name__ == "__main__":
    unittest.main()
