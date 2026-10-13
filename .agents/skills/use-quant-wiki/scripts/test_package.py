#!/usr/bin/env python3
"""Meaningful package lifecycle checks; fixtures stay inside this repository."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOL = Path(__file__).with_name("package.py")
if not TOOL.is_file():
    raise RuntimeError("The test requires package.py in the same directory")
WORKSPACE = Path.cwd().resolve()


class PackageLifecycleTests(unittest.TestCase):
    def setUp(self):
        if not (WORKSPACE / "AGENTS.md").is_file() or not (WORKSPACE / ".git").exists():
            self.fail("Run these checks from the quantization repository root")
        self.temp = tempfile.TemporaryDirectory(prefix=".package-tool-test-", dir=WORKSPACE)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source, self.meta = self.make_source("source")
        self.project = self.base / "new-project"
        self.installation = self.project / ".agents" / "skills" / "use-quant-wiki"
        self.publish()
        spec = importlib.util.spec_from_file_location("package_tool", TOOL)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def make_source(self, name):
        source = self.base / name
        (source / "scripts").mkdir(parents=True)
        (source / "skills" / "wiki-query").mkdir(parents=True)
        shutil.copy2(TOOL, source / "scripts" / "package.py")
        for path, content in {"README.md": "Package design\n", "SKILL.md": "Entry version one\n",
                              "skills/wiki-query/README.md": "Query design\n",
                              "skills/wiki-query/SKILL.md": "Query execution\n"}.items():
            (source / path).write_text(content, encoding="utf-8")
        wiki = self.base / (name + "-wiki")
        wiki.mkdir()
        metadata = {"schema_version": 1, "name": "use-quant-wiki",
                    "bindings": {"wiki_root": str(wiki), "source_package": str(source),
                                 "feedback_root": str(wiki / "query-feedback")},
                    "capabilities": [{"name": "wiki-query", "purpose": "Use knowledge", "path": "skills/wiki-query/SKILL.md"}]}
        meta = self.base / (name + "-metadata.json")
        meta.write_text(json.dumps(metadata), encoding="utf-8")
        return source, meta

    def call(self, *arguments, package=None, expected=0, default_tool=False):
        package = package or self.source
        if default_tool:
            command = [sys.executable, str(package / "scripts/package.py"), *arguments]
        else:
            command = [sys.executable, str(TOOL), "--package", str(package), *arguments]
        completed = subprocess.run(command, cwd=WORKSPACE, text=True, capture_output=True)
        self.assertEqual(completed.returncode, expected, completed.stdout + completed.stderr)
        self.assertEqual(completed.stderr, "", completed.stderr)
        return json.loads(completed.stdout)

    def publish(self, version=1, source=None, meta=None):
        return self.call("publish", "--version", str(version), "--metadata", str(meta or self.meta), package=source or self.source)

    def install(self, **kwargs):
        return self.call("install", "--project", str(self.project), **kwargs)

    def feedback_input(self, **changes):
        value = {"task_id": "task-1", "task_summary": "Explain a deployment choice",
                 "capabilities_used": ["wiki-query"], "knowledge_locations": [],
                 "result_summary": "A bounded answer", "limitations": [], "suggestions": []}
        value.update(changes)
        path = self.base / "feedback-input.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_actual_install_creates_project_and_idempotent_reuse(self):
        self.assertFalse(self.project.exists())
        result = self.install()
        self.assertEqual(result["state"], "installed")
        self.assertTrue((self.installation / "skills/wiki-query/README.md").is_file())
        self.assertEqual(self.call("verify", package=self.installation, default_tool=True)["state"], "valid")
        tracked = {path.relative_to(self.installation): (path.read_bytes(), path.stat().st_mtime_ns)
                   for path in self.installation.rglob("*") if path.is_file()}
        self.assertEqual(self.install()["state"], "reused")
        self.assertEqual(tracked, {path.relative_to(self.installation): (path.read_bytes(), path.stat().st_mtime_ns)
                                  for path in self.installation.rglob("*") if path.is_file()})
        self.assertEqual(self.call("status", package=self.installation)["state"], "current")

    def test_update_keeps_task_complete_snapshot_at_old_version(self):
        self.install()
        output = self.base / "tasks/task-1/package"
        self.assertEqual(self.call("snapshot", "--output", str(output), package=self.installation, default_tool=True)["state"], "snapshot_created")
        old_entry = (output / "SKILL.md").read_bytes()
        (self.source / "SKILL.md").write_text("Entry version two\n", encoding="utf-8")
        self.publish(2)
        self.assertEqual(self.call("status", package=self.installation)["state"], "outdated")
        self.assertEqual(self.install(expected=2)["state"], "update_required")
        self.assertEqual(self.call("install", "--project", str(self.project), "--allow-update")["state"], "updated")
        self.assertNotEqual((self.installation / "SKILL.md").read_bytes(), old_entry)
        self.assertEqual((output / "SKILL.md").read_bytes(), old_entry)
        self.assertEqual(self.call("verify", package=output, default_tool=True)["version"], 1)
        self.assertEqual(self.call("status", package=output, default_tool=True)["state"], "fixed_snapshot")
        self.assertEqual(self.call("snapshot", "--output", str(output), expected=2)["state"], "snapshot_exists")

    def test_customized_payload_and_metadata_cannot_be_overwritten(self):
        self.install()
        entry = self.installation / "SKILL.md"
        entry.write_text("Project-specific change\n", encoding="utf-8")
        self.assertEqual(self.call("status", package=self.installation)["state"], "customized")
        self.assertEqual(self.call("install", "--project", str(self.project), "--allow-update", expected=2)["state"], "customized")
        self.assertEqual(entry.read_text(), "Project-specific change\n")
        shutil.copy2(self.source / "SKILL.md", entry)
        manifest_path = self.installation / "package.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["version"] = 7
        body = {key: value for key, value in manifest.items() if key != "content_sha256"}
        manifest["content_sha256"] = self.module.sha256(self.module.canonical(body))
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(self.call("verify", package=self.installation, expected=2)["state"], "customized")
        self.assertEqual(self.call("install", "--project", str(self.project), "--allow-update", expected=2)["state"], "customized")
        self.assertEqual(json.loads(manifest_path.read_text())["version"], 7)

    def test_other_source_and_other_skills_are_preserved(self):
        self.install()
        other = self.project / ".agents/skills/keep-me/SKILL.md"
        other.parent.mkdir()
        other.write_text("Other skill\n")
        before = (self.installation / "package.json").read_bytes()
        source2, meta2 = self.make_source("other-source")
        self.publish(source=source2, meta=meta2)
        result = self.call("install", "--project", str(self.project), "--source", str(source2), "--allow-update", expected=2)
        self.assertEqual(result["state"], "different_source")
        self.assertEqual((self.installation / "package.json").read_bytes(), before)
        self.assertEqual(other.read_text(), "Other skill\n")

    def test_invalid_source_does_not_touch_new_or_existing_target(self):
        self.install()
        before = (self.installation / "package.json").read_bytes()
        (self.source / "SKILL.md").write_text("Unpublished working change\n")
        self.assertEqual(self.install(expected=2)["state"], "source_modified")
        self.assertEqual((self.installation / "package.json").read_bytes(), before)
        missing = self.base / "not-created"
        result = self.call("install", "--project", str(missing), expected=2)
        self.assertEqual(result["state"], "source_modified")
        self.assertFalse(missing.exists())
        self.assertEqual(self.call("publish", "--version", "1", expected=2)["state"], "version_not_increasing")
        unpublished, _ = self.make_source("unpublished")
        result = self.call("install", "--project", str(missing), "--source", str(unpublished), expected=2)
        self.assertEqual(result["state"], "unpublished")
        self.assertFalse(missing.exists())

    def test_changed_target_during_prepare_is_preserved(self):
        self.install()
        (self.source / "SKILL.md").write_text("Source version two\n")
        self.publish(2)
        original_copy = self.module.copy_payload
        def edit_target_after_copy(source, destination, manifest):
            original_copy(source, destination, manifest)
            (self.installation / "SKILL.md").write_text("Concurrent project edit\n")
        with mock.patch.object(self.module, "copy_payload", side_effect=edit_target_after_copy):
            with self.assertRaises(self.module.PackageError) as caught:
                self.module.install(self.source, self.project, allow_update=True)
        self.assertEqual(caught.exception.state, "target_changed")
        self.assertEqual((self.installation / "SKILL.md").read_text(), "Concurrent project edit\n")
        self.assertEqual(json.loads((self.installation / "package.json").read_text())["version"], 1)
        self.assertFalse(list(self.installation.parent.glob(".use-quant-wiki-stage-*")))

    def test_failed_promotion_restores_previous_complete_installation(self):
        self.install()
        before = (self.installation / "package.json").read_bytes()
        (self.source / "SKILL.md").write_text("Source version two\n")
        self.publish(2)
        real_replace = self.module.os.replace
        def fail_promotion(source, target):
            if "-stage-" in str(source) and Path(target) == self.installation:
                raise OSError("Injected rename failure")
            return real_replace(source, target)
        with mock.patch.object(self.module.os, "replace", side_effect=fail_promotion):
            with self.assertRaises(OSError):
                self.module.install(self.source, self.project, allow_update=True)
        self.assertEqual((self.installation / "package.json").read_bytes(), before)
        self.assertEqual(self.call("verify", package=self.installation)["version"], 1)
        self.assertFalse(list(self.installation.parent.glob(".use-quant-wiki-backup-*")))

    def test_intact_mirror_and_snapshot_work_when_source_unavailable(self):
        self.install()
        self.source.rename(self.base / "offline-source")
        observation = self.call("status", package=self.installation, default_tool=True)
        self.assertEqual(observation["state"], "installed_source_unavailable")
        self.assertEqual(observation["source_state"], "unavailable")
        frozen = self.base / "offline-task/package"
        self.call("snapshot", "--output", str(frozen), package=self.installation, default_tool=True)
        self.assertEqual(self.call("verify", package=frozen, default_tool=True)["state"], "valid")

    def test_final_readback_failure_restores_old_package_and_retains_new(self):
        self.install()
        before = (self.installation / "package.json").read_bytes()
        (self.source / "SKILL.md").write_text("Source version two\n")
        self.publish(2)
        real_verify = self.module.valid_for_use
        def fail_only_final_target(path):
            if path == self.installation and json.loads((path / "package.json").read_text())["version"] == 2:
                raise self.module.PackageError("source_modified", "Injected final readback failure")
            return real_verify(path)
        with mock.patch.object(self.module, "valid_for_use", side_effect=fail_only_final_target):
            with self.assertRaises(self.module.PackageError) as caught:
                self.module.install(self.source, self.project, allow_update=True)
        self.assertEqual(caught.exception.state, "installation_verification_failed")
        self.assertTrue(caught.exception.details["restored_previous"])
        self.assertEqual((self.installation / "package.json").read_bytes(), before)
        retained = Path(caught.exception.details["retained_failed_package"])
        self.assertEqual(json.loads((retained / "package.json").read_text())["version"], 2)
        self.assertEqual(self.call("verify", package=self.installation)["version"], 1)

    def test_first_install_final_readback_failure_reports_actual_directory(self):
        real_verify = self.module.valid_for_use
        def fail_only_final_target(path):
            if path == self.installation:
                raise self.module.PackageError("unavailable", "Injected first-install read failure")
            return real_verify(path)
        with mock.patch.object(self.module, "valid_for_use", side_effect=fail_only_final_target):
            with self.assertRaises(self.module.PackageError) as caught:
                self.module.install(self.source, self.project)
        self.assertEqual(caught.exception.state, "installation_verification_failed")
        self.assertFalse(caught.exception.details["restored_previous"])
        self.assertEqual(caught.exception.details["installation"], str(self.installation))
        self.assertTrue((self.installation / "package.json").is_file())

    def test_first_feedback_creates_only_bound_directory_and_unique_files(self):
        feedback_root = self.base / "source-wiki/query-feedback"
        self.assertFalse(feedback_root.exists())
        data = self.feedback_input()
        first = self.call("feedback", "--input", str(data))
        self.assertEqual(first["state"], "feedback_written")
        first_path = Path(first["feedback"])
        before = first_path.read_bytes()
        second = self.call("feedback", "--input", str(data))
        self.assertNotEqual(first["query_id"], second["query_id"])
        self.assertEqual(first_path.read_bytes(), before)
        self.assertEqual(len(list(feedback_root.iterdir())), 2)
        self.assertEqual(json.loads(before)["package_version"], 1)
        self.assertEqual(self.call("verify")["state"], "valid")

    def test_invalid_feedback_or_missing_wiki_does_not_create_logs(self):
        folder = self.base / "source-wiki/query-feedback"
        for changes in ({"knowledge_locations": ["wiki/INDEX.md"]}, {"capabilities_used": []}, {"task_summary": ""}):
            data = self.feedback_input(**changes)
            result = self.call("feedback", "--input", str(data), expected=2)
            self.assertIn(result["state"], {"invalid_feedback", "invalid_path"})
            self.assertFalse(folder.exists())
        (self.base / "source-wiki").rmdir()
        self.assertEqual(self.call("feedback", "--input", str(self.feedback_input()), expected=2)["state"], "feedback_unavailable")
        self.assertFalse((self.base / "source-wiki").exists())

    def test_url_symlink_and_path_traversal_are_refused(self):
        self.assertEqual(self.call("install", "--project", "https://example.org/repository", expected=2)["state"], "invalid_path")
        actual = self.base / "unrelated"
        actual.mkdir()
        alias = self.base / "alias"
        alias.symlink_to(actual, target_is_directory=True)
        self.assertEqual(self.call("install", "--project", str(alias), expected=2)["state"], "unsafe_target")
        self.assertEqual(list(actual.iterdir()), [])
        manifest = json.loads((self.source / "package.json").read_text())
        manifest["files"]["../outside"] = "0" * 64
        (self.source / "package.json").write_text(json.dumps(manifest))
        self.assertEqual(self.call("verify", expected=2)["state"], "invalid_manifest")

    def test_required_internal_design_and_manifest_tampering(self):
        (self.source / "skills/wiki-query/README.md").unlink()
        self.assertEqual(self.call("publish", "--version", "2", expected=2)["state"], "incomplete_package")
        (self.source / "skills/wiki-query/README.md").write_text("Query design\n")
        manifest = json.loads((self.source / "package.json").read_text())
        manifest["capabilities"][0]["purpose"] = "Changed metadata without release"
        (self.source / "package.json").write_text(json.dumps(manifest))
        self.assertEqual(self.call("verify", expected=2)["state"], "invalid_manifest")


if __name__ == "__main__":
    unittest.main(verbosity=2)
