"""Source-only release inventory and real-path privacy rejection."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("source_builder", ROOT / "scripts/build_public_toolkit.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PackageTests(unittest.TestCase):
    def test_actual_home_path_rejected_but_prefix_regex_is_not_a_path(self):
        actual = b"/" + b"Users" + b"/" + b"PRIVATE_TEST_USER/document.txt"
        self.assertTrue(builder.has_private_home_path(actual))
        self.assertFalse(builder.has_private_home_path(rb"/Users/|/home/"))
        self.assertTrue(builder.has_private_home_path(b"/" + b"home" + b"/private_test/data"))

    def test_public_demo_is_complete_in_source_candidate(self):
        included = {p.relative_to(ROOT).as_posix() for p in builder.source_files()}
        expected = {p.relative_to(ROOT).as_posix() for p in (ROOT / "public_demo").rglob("*") if p.is_file()}
        self.assertTrue(expected <= included)

    def test_sensitive_runtime_folders_and_local_config_excluded(self):
        files = {p.relative_to(ROOT).as_posix() for p in builder.source_files()}
        self.assertNotIn("recognition.local.json", files)
        self.assertFalse(any(p.startswith(("local_workspace/", "results/", "dist/", ".git/")) for p in files))


if __name__ == "__main__":
    unittest.main()
