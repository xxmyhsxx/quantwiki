#!/usr/bin/env python3
"""Package lifecycle regressions; all fixtures stay in private temporary directories."""
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
        self.temp = tempfile.TemporaryDirectory(prefix="wiki-package-test-", dir="/private/tmp")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.package_name = "use-demo-wiki"
        self.source, self.meta = self.make_source("source")
        self.project = self.base / "new-project"
        self.installation = self.project / ".agents" / "skills" / self.package_name
        self.publish()
        spec = importlib.util.spec_from_file_location("package_tool", TOOL)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def make_source(self, name):
        source = self.base / name / self.package_name
        (source / "scripts").mkdir(parents=True)
        (source / "skills" / "wiki-query").mkdir(parents=True)
        shutil.copy2(TOOL, source / "scripts" / "package.py")
        for path, content in {"README.md": "Package design\n", "SKILL.md": "Entry version one\n",
                              "skills/wiki-query/README.md": "Query design\n",
                              "skills/wiki-query/SKILL.md": "Query execution\n"}.items():
            (source / path).write_text(content, encoding="utf-8")
        wiki = self.base / (name + "-wiki")
        wiki.mkdir()
        metadata = {"schema_version": 1, "name": self.package_name,
                    "bindings": {"wiki_root": str(wiki), "source_package": str(source),
                                 "feedback_root": str(wiki / "query-feedback")},
                    "capabilities": [{"name": "wiki-query", "purpose": "Use knowledge", "path": "skills/wiki-query/SKILL.md"}]}
        meta = self.base / (name + "-metadata.json")
        meta.write_text(json.dumps(metadata), encoding="utf-8")
        return source, meta

    def call(self, *arguments, package=None, expected=0, default_tool=False):
        package = package or self.source
        if default_tool:
            command = [sys.executable, "-B", str(package / "scripts/package.py"), *arguments]
        else:
            command = [sys.executable, "-B", str(TOOL), "--package", str(package), *arguments]
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
        self.assertFalse(list(self.installation.parent.glob(f".{self.package_name}-stage-*")))

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
        self.assertFalse(list(self.installation.parent.glob(f".{self.package_name}-backup-*")))

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
        def fail_only_final_target(path, **kwargs):
            if path == self.installation and json.loads((path / "package.json").read_text())["version"] == 2:
                raise self.module.PackageError("source_modified", "Injected final readback failure")
            return real_verify(path, **kwargs)
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
        def fail_only_final_target(path, **kwargs):
            if path == self.installation:
                raise self.module.PackageError("unavailable", "Injected first-install read failure")
            return real_verify(path, **kwargs)
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







    def test_dynamic_name_and_complete_selected_dependencies(self):
        self.package_name = "use-policy-library"
        source, meta = self.make_source("policy")
        helper = source / "skills" / "compare-policy"
        helper.mkdir()
        (helper / "README.md").write_text("Comparison design\n")
        (helper / "SKILL.md").write_text("Read references/example.md\n")
        (helper / "references").mkdir()
        dependency = helper / "references/example.md"
        dependency.write_text("Actual selected dependency\n")
        data = json.loads(meta.read_text())
        data["capabilities"].append({"name": "compare-policy", "purpose": "Compare policies", "path": "skills/compare-policy/SKILL.md"})
        meta.write_text(json.dumps(data))
        self.publish(source=source, meta=meta)
        project = self.base / "policy-project"
        result = self.call("install", "--project", str(project), "--source", str(source))
        target = project / ".agents/skills" / self.package_name
        self.assertEqual(result["installation"], str(target))
        manifest = json.loads((target / "package.json").read_text())
        self.assertEqual(manifest["name"], self.package_name)
        self.assertEqual(len(manifest["capabilities"]), 2)
        self.assertEqual((target / "skills/compare-policy/references/example.md").read_bytes(), dependency.read_bytes())
        self.assertEqual(self.call("verify", package=target, default_tool=True)["state"], "valid")

    def test_unpublished_template_requires_initial_metadata(self):
        source, meta = self.make_source("unpublished-template")
        self.assertEqual(self.call("verify", package=source, expected=2)["state"], "unpublished")
        self.assertEqual(self.call("status", package=source)["state"], "unpublished")
        self.assertEqual(self.call("publish", "--version", "1", package=source, expected=2)["state"], "unpublished")
        self.assertFalse((source / "package.json").exists())
        self.assertEqual(self.publish(source=source, meta=meta)["state"], "published")
        self.assertEqual(self.call("publish", "--version", "1", package=source)["state"], "already_published")

    def test_invalid_dynamic_name_and_source_binding_are_refused(self):
        source, meta = self.make_source("invalid-name")
        data = json.loads(meta.read_text())
        for name in ("../escape", "UpperCase", "with_under", "-prefix", "two--hyphens", "a" * 65):
            changed = {**data, "name": name}
            meta.write_text(json.dumps(changed))
            self.assertEqual(self.call("publish", "--version", "1", "--metadata", str(meta), package=source, expected=2)["state"], "invalid_manifest")
            self.assertFalse((source / "package.json").exists())
        data["bindings"]["source_package"] = str(source.parent / "other-name")
        meta.write_text(json.dumps(data))
        self.assertEqual(self.call("publish", "--version", "1", "--metadata", str(meta), package=source, expected=2)["state"], "invalid_manifest")

    def test_late_edit_after_final_comparison_survives_in_backup(self):
        self.install()
        (self.source / "SKILL.md").write_text("Source version two\n")
        self.publish(2)
        real_replace = self.module.os.replace
        def edit_just_before_old_directory_move(source, target):
            if Path(source) == self.installation and "-backup-" in Path(target).name:
                (self.installation / "SKILL.md").write_text("Late project customization\n")
            return real_replace(source, target)
        with mock.patch.object(self.module.os, "replace", side_effect=edit_just_before_old_directory_move):
            result = self.module.install(self.source, self.project, allow_update=True)
        self.assertEqual(result["state"], "updated")
        backup = Path(result["previous_installation"])
        self.assertTrue(backup.is_dir())
        self.assertEqual((backup / "SKILL.md").read_text(), "Late project customization\n")
        self.assertEqual((self.installation / "SKILL.md").read_text(), "Source version two\n")
        self.assertEqual(self.call("verify", package=self.installation)["version"], 2)

    def test_current_copy_to_new_project_requires_explicit_record_refresh(self):
        self.install()
        project_b = self.base / "project-b"
        copied = project_b / ".agents/skills" / self.package_name
        shutil.copytree(self.installation, copied)
        previous = (copied / ".installation.json").read_bytes()
        self.assertEqual(self.call("verify", package=copied, expected=2)["state"], "installation_mismatch")
        self.assertEqual(self.call("status", package=copied)["state"], "installation_mismatch")
        result = self.call("install", "--project", str(project_b))
        self.assertTrue(result["installation_record_refreshed"])
        record = json.loads((copied / ".installation.json").read_text())
        self.assertEqual(record["project_root"], str(project_b))
        self.assertEqual(record["skills_root"], str(copied.parent))
        self.assertEqual(record["scope"], "project")
        self.assertNotEqual((copied / ".installation.json").read_bytes(), previous)
        self.assertEqual(self.call("verify", package=copied)["state"], "valid")

    def test_old_copied_installation_is_not_adopted_or_updated(self):
        self.install()
        project_b = self.base / "old-copy-project"
        copied = project_b / ".agents/skills" / self.package_name
        shutil.copytree(self.installation, copied)
        before = {p.relative_to(copied): p.read_bytes() for p in copied.rglob("*") if p.is_file()}
        (self.source / "SKILL.md").write_text("Source version two\n")
        self.publish(2)
        self.assertEqual(self.call("install", "--project", str(project_b), "--allow-update", expected=2)["state"], "installation_mismatch")
        self.assertEqual(before, {p.relative_to(copied): p.read_bytes() for p in copied.rglob("*") if p.is_file()})
        self.source.rename(self.base / "offline-source")
        self.assertEqual(self.call("install", "--project", str(project_b), expected=2)["state"], "unavailable")
        self.assertEqual(before, {p.relative_to(copied): p.read_bytes() for p in copied.rglob("*") if p.is_file()})

    def test_shared_directory_installation_and_copied_record_refresh(self):
        shared = self.base / "shared-a"
        result = self.call("install", "--skills-dir", str(shared))
        installed = shared / self.package_name
        self.assertEqual(result["scope"], "shared")
        self.assertIsNone(result["project"])
        record = json.loads((installed / ".installation.json").read_text())
        self.assertEqual(record["skills_root"], str(shared))
        self.assertIsNone(record["project_root"])
        self.assertEqual(self.call("verify", package=installed)["state"], "valid")
        shared_b = self.base / "shared-b"
        copied = shared_b / self.package_name
        shutil.copytree(installed, copied)
        self.assertEqual(self.call("status", package=copied)["state"], "installation_mismatch")
        self.assertTrue(self.call("install", "--skills-dir", str(shared_b))["installation_record_refreshed"])
        self.assertEqual(json.loads((copied / ".installation.json").read_text())["skills_root"], str(shared_b))
        project_b = self.base / "shared-to-project"
        project_copy = project_b / ".agents/skills" / self.package_name
        shutil.copytree(copied, project_copy)
        self.call("install", "--project", str(project_b))
        self.assertEqual(json.loads((project_copy / ".installation.json").read_text())["scope"], "project")
        self.assertEqual(self.call("verify", package=project_copy)["state"], "valid")

    def test_feedback_symlink_and_non_directory_targets_do_not_escape(self):
        folder = self.base / "source-wiki/query-feedback"
        outside = self.base / "unrelated-output"
        outside.mkdir()
        folder.symlink_to(outside, target_is_directory=True)
        result = self.call("feedback", "--input", str(self.feedback_input()), expected=2)
        self.assertIn(result["state"], {"invalid_manifest", "feedback_unavailable"})
        self.assertEqual(list(outside.iterdir()), [])
        folder.unlink()
        folder.write_text("Preserve this file\n")
        self.assertEqual(self.call("feedback", "--input", str(self.feedback_input()), expected=2)["state"], "feedback_unavailable")
        self.assertEqual(folder.read_text(), "Preserve this file\n")

    def test_copied_custom_record_is_preserved_instead_of_refreshed(self):
        self.install()
        record_path = self.installation / ".installation.json"
        value = json.loads(record_path.read_text())
        value["user_note"] = "Preserve receiving-side customization"
        record_path.write_text(json.dumps(value))
        for option, receiving in (("--project", self.base / "custom-copy-project"),
                                  ("--skills-dir", self.base / "custom-copy-shared")):
            target = receiving / ".agents/skills" / self.package_name if option == "--project" else receiving / self.package_name
            shutil.copytree(self.installation, target)
            before = (target / ".installation.json").read_bytes()
            self.assertEqual(self.call("install", option, str(receiving), expected=2)["state"], "customized")
            self.assertEqual((target / ".installation.json").read_bytes(), before)
            self.assertEqual(json.loads(before)["user_note"], value["user_note"])
        value.pop("user_note")
        value["scope"] = []
        record_path.write_text(json.dumps(value))
        self.assertEqual(self.call("verify", package=self.installation, expected=2)["state"], "customized")

    def test_prepare_source_creates_one_verified_complete_snapshot(self):
        output = self.base / "tasks/source/package"
        real_scan = self.module.payload_files
        scans = []
        def track_scan(path):
            scans.append(path)
            return real_scan(path)
        with mock.patch.object(self.module, "payload_files", side_effect=track_scan):
            result = self.module.prepare(self.source, output)
        self.assertEqual(result["state"], "ready")
        self.assertEqual(result["snapshot"], str(output))
        self.assertEqual(result["freshness"], "current")
        self.assertTrue(result["source_checked"])
        self.assertFalse(result["updated"])
        self.assertEqual(result["limitations"], [])
        self.assertEqual(scans.count(self.source), 1)
        self.assertEqual(scans.count(output), 1)
        self.assertEqual(len(scans), 3)
        self.assertTrue((output / "skills/wiki-query/SKILL.md").is_file())
        self.assertFalse((self.source / ".installation.json").exists())
        self.assertFalse(self.project.exists())

    def test_prepare_uses_old_release_without_update_permission_then_keeps_backup(self):
        self.install()
        before = {p.relative_to(self.installation): p.read_bytes()
                  for p in self.installation.rglob("*") if p.is_file()}
        (self.source / "SKILL.md").write_text("Source version two\n")
        self.publish(2)
        old_task = self.base / "tasks/old/package"
        initial = self.call("prepare", "--output", str(old_task), package=self.installation, default_tool=True)
        self.assertEqual(initial["state"], "ready")
        self.assertEqual(initial["version"], 1)
        self.assertEqual(initial["freshness"], "outdated")
        self.assertFalse(initial["updated"])
        self.assertTrue(initial["limitations"])
        self.assertEqual(before, {p.relative_to(self.installation): p.read_bytes()
                                 for p in self.installation.rglob("*") if p.is_file()})
        new_task = self.base / "tasks/new/package"
        result = self.call("prepare", "--output", str(new_task), "--allow-update", package=self.installation, default_tool=True)
        self.assertEqual(result["version"], 2)
        self.assertEqual(result["freshness"], "current")
        self.assertTrue(result["updated"])
        self.assertEqual(result["limitations"], [])
        backup = Path(result["previous_installation"])
        self.assertTrue(backup.is_dir())
        self.assertEqual(before, {p.relative_to(backup): p.read_bytes()
                                 for p in backup.rglob("*") if p.is_file()})
        self.assertEqual(json.loads((old_task / "package.json").read_text())["version"], 1)
        self.assertEqual(initial["content_sha256"], json.loads((old_task / "package.json").read_text())["content_sha256"])

    def test_prepare_intact_mirror_with_unavailable_or_draft_source(self):
        self.install()
        (self.source / "SKILL.md").write_text("Unpublished work\n")
        draft = self.call("prepare", "--output", str(self.base / "draft/package"), package=self.installation)
        self.assertEqual(draft["freshness"], "source_unpublished")
        self.assertTrue(draft["source_checked"])
        self.assertFalse(draft["updated"])
        self.assertTrue(draft["limitations"])
        self.source.rename(self.base / "offline-source")
        offline = self.call("prepare", "--output", str(self.base / "offline/package"), package=self.installation)
        self.assertEqual(offline["state"], "ready")
        self.assertEqual(offline["freshness"], "source_unavailable")
        self.assertFalse(offline["source_checked"])
        self.assertEqual(offline["version"], 1)
        self.assertTrue(offline["limitations"])

    def test_prepare_fixed_snapshot_reuses_it_without_source_access(self):
        output = self.base / "fixed/package"
        self.call("prepare", "--output", str(output))
        unused = self.base / "must-not-create/package"
        before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
        with mock.patch.object(self.module, "inspect", side_effect=AssertionError("Fixed task must not inspect source")):
            result = self.module.prepare(output, unused, allow_update=True)
        self.assertEqual(result["snapshot"], str(output))
        self.assertEqual(result["freshness"], "fixed_snapshot")
        self.assertFalse(result["source_checked"])
        self.assertFalse(unused.exists())
        self.assertEqual(before, {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()})

    def test_prepare_refuses_customization_source_conflict_and_downgrade(self):
        self.install()
        entry = self.installation / "SKILL.md"
        original = entry.read_bytes()
        entry.write_text("Local task method\n")
        output = self.base / "refused/package"
        refusal = self.call("prepare", "--output", str(output), "--allow-update", package=self.installation, expected=2)
        self.assertEqual(refusal["state"], "customized")
        self.assertFalse(output.exists())
        self.assertEqual(entry.read_text(), "Local task method\n")
        entry.write_bytes(original)
        manifest_path = self.source / "package.json"
        original_manifest = manifest_path.read_bytes()
        other_origin = json.loads(original_manifest)
        other_origin["bindings"]["wiki_root"] = str(self.base / "other-wiki")
        other_origin["bindings"]["feedback_root"] = str(self.base / "other-wiki/query-feedback")
        other_origin["content_sha256"] = self.module.sha256(self.module.canonical({key: value for key, value in other_origin.items() if key != "content_sha256"}))
        manifest_path.write_text(json.dumps(other_origin))
        self.assertEqual(self.call("prepare", "--output", str(output), "--allow-update", package=self.installation, expected=2)["state"], "different_source")
        manifest_path.write_bytes(original_manifest)
        (self.source / "SKILL.md").write_text("Same-version conflicting publication\n")
        current = json.loads(original_manifest)
        current["files"] = self.module.payload_files(self.source)
        current["content_sha256"] = self.module.sha256(self.module.canonical({key: value for key, value in current.items() if key != "content_sha256"}))
        manifest_path.write_text(json.dumps(current))
        self.assertEqual(self.call("prepare", "--output", str(output), "--allow-update", package=self.installation, expected=2)["state"], "release_conflict")
        (self.source / "SKILL.md").write_bytes(original)
        manifest_path.write_bytes(original_manifest)
        self.publish(2)
        self.call("install", "--project", str(self.project), "--allow-update")
        manifest_path.write_bytes(original_manifest)
        self.assertEqual(self.call("prepare", "--output", str(output), "--allow-update", package=self.installation, expected=2)["state"], "ahead")
        self.assertFalse(output.exists())
        self.assertEqual(json.loads((self.installation / "package.json").read_text())["version"], 2)

    def test_prepare_rejects_invalid_or_unsafe_source_without_using_it(self):
        self.install()
        manifest_path = self.source / "package.json"
        original = manifest_path.read_bytes()
        manifest_path.write_text("{broken")
        output = self.base / "untrusted/package"
        before = (self.installation / ".installation.json").read_bytes()
        self.assertEqual(self.call("prepare", "--output", str(output), "--allow-update", package=self.installation, expected=2)["state"], "invalid_manifest")
        self.assertFalse(output.exists())
        self.assertEqual((self.installation / ".installation.json").read_bytes(), before)
        manifest_path.write_bytes(original)
        target = self.base / "unrelated"
        target.write_text("Must not be loaded\n")
        (self.source / "unsafe-link").symlink_to(target)
        self.assertEqual(self.call("prepare", "--output", str(output), package=self.installation, expected=2)["state"], "unsafe_payload")
        self.assertFalse(output.exists())
        self.assertEqual(target.read_text(), "Must not be loaded\n")

    def test_prepare_shared_update_checks_source_once_and_keeps_final_verification(self):
        shared = self.base / "shared"
        self.call("install", "--skills-dir", str(shared))
        installed = shared / self.package_name
        (self.source / "SKILL.md").write_text("New shared release\n")
        self.publish(2)
        real_scan = self.module.payload_files
        scans = []
        def track_scan(path):
            scans.append(path)
            return real_scan(path)
        output = self.base / "shared-task/package"
        with mock.patch.object(self.module, "payload_files", side_effect=track_scan):
            result = self.module.prepare(installed, output, allow_update=True)
        self.assertTrue(result["updated"])
        self.assertEqual(result["version"], 2)
        self.assertTrue(Path(result["previous_installation"]).is_dir())
        self.assertEqual(scans.count(self.source), 1)
        self.assertEqual(scans.count(installed), 3)
        self.assertEqual(scans.count(output), 1)
        self.assertEqual(json.loads((installed / ".installation.json").read_text())["scope"], "shared")

    def test_prepare_reports_retained_backup_when_snapshot_fails_after_update(self):
        self.install()
        (self.source / "SKILL.md").write_text("Source version two\n")
        self.publish(2)
        occupied = self.base / "existing-snapshot"
        occupied.mkdir()
        (occupied / "user-file").write_text("Keep this result\n")
        result = self.call("prepare", "--output", str(occupied), "--allow-update", package=self.installation, expected=2)
        self.assertEqual(result["state"], "snapshot_exists")
        self.assertTrue(result["updated"])
        self.assertEqual(result["version"], 2)
        self.assertTrue(Path(result["previous_installation"]).is_dir())
        self.assertEqual((occupied / "user-file").read_text(), "Keep this result\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
