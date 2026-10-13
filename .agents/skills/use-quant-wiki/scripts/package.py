#!/usr/bin/env python3
"""Publish, inspect, install and freeze a complete Wiki capability package.

Only Python's standard library is required. Every command returns JSON. Refused
writes preserve the target; post-install verification failures report rollback
and any retained artifacts rather than claiming the target was never changed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import uuid

MANIFEST = "package.json"
INSTALLATION = ".installation.json"
SNAPSHOT = ".snapshot.json"
RECORDS = {MANIFEST, INSTALLATION, SNAPSHOT}
SCHEMA = 1
DEFAULT_PACKAGE = Path(__file__).resolve().parent.parent


class PackageError(Exception):
    def __init__(self, state, message, **details):
        super().__init__(message)
        self.state, self.details = state, details


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def timestamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8), "Asia/Taipei")).isoformat(timespec="seconds")


def absolute(value, label):
    if not isinstance(value, str) or not Path(value).is_absolute() or "://" in value:
        raise PackageError("invalid_path", f"{label} must be an absolute local filesystem path", path=value)
    return Path(value).resolve()


def destination(value, label):
    absolute(value, label)
    path = Path(value)
    for component in (path, *path.parents):
        if component.is_symlink():
            raise PackageError("unsafe_target", "Destination paths must not contain symbolic links", path=str(component))
    return path.resolve()


def read_json(path, state="invalid_manifest"):
    try:
        if path.is_symlink():
            raise PackageError(state, "JSON file must not be a symbolic link", path=str(path))
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except PackageError:
        raise
    except (OSError, ValueError) as exc:
        raise PackageError(state, str(exc), path=str(path)) from exc


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_json(path, value):
    """Replace one generated JSON file only after writing its complete contents."""
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise PackageError("invalid_manifest", "Payload path must be a POSIX relative path", path=value)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in value.split("/") or str(path) != value:
        raise PackageError("invalid_manifest", "Unsafe or non-canonical payload path", path=value)
    if value in RECORDS or "__pycache__" in path.parts or value.endswith((".pyc", ".pyo")):
        raise PackageError("invalid_manifest", "Generated records and bytecode are not payload", path=value)
    return path


def skill_name(value):
    if not isinstance(value, str) or len(value) > 64 or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
        raise PackageError("invalid_manifest", "Names must be safe Skill names of 1-64 lowercase letters, digits or single hyphens", name=value)
    return value


def metadata(value, version=None):
    if not isinstance(value, dict):
        raise PackageError("invalid_manifest", "Metadata must be a JSON object")
    if value.get("schema_version") != SCHEMA:
        raise PackageError("invalid_manifest", "Unsupported schema_version")
    package_name = skill_name(value.get("name"))
    chosen = value.get("version") if version is None else version
    if type(chosen) is not int or chosen < 1:
        raise PackageError("invalid_manifest", "version must be a positive integer")
    bindings = value.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != {"wiki_root", "source_package", "feedback_root"}:
        raise PackageError("invalid_manifest", "bindings requires wiki_root, source_package and feedback_root")
    resolved = {key: str(absolute(path, key)) for key, path in bindings.items()}
    if resolved != bindings:
        raise PackageError("invalid_manifest", "Bindings must use canonical absolute paths")
    if Path(resolved["source_package"]).name != package_name:
        raise PackageError("invalid_manifest", "source_package directory must match the package name")
    if Path(resolved["feedback_root"]) != Path(resolved["wiki_root"]) / "query-feedback":
        raise PackageError("invalid_manifest", "feedback_root must be wiki_root/query-feedback")
    capabilities = value.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        raise PackageError("invalid_manifest", "capabilities must be a non-empty list")
    seen = set()
    for capability in capabilities:
        if not isinstance(capability, dict) or set(capability) != {"name", "purpose", "path"}:
            raise PackageError("invalid_manifest", "Each capability requires name, purpose and path")
        name, purpose = capability["name"], capability["purpose"]
        skill_name(name)
        if name in seen:
            raise PackageError("invalid_manifest", "Capability names must be unique")
        if not isinstance(purpose, str) or not purpose.strip():
            raise PackageError("invalid_manifest", "Capability purpose must not be empty")
        path = relative(capability["path"])
        if path != PurePosixPath("skills") / name / "SKILL.md":
            raise PackageError("invalid_manifest", "Capability path must be skills/<name>/SKILL.md")
        seen.add(name)
    return {"schema_version": SCHEMA, "name": package_name, "version": chosen,
            "bindings": bindings, "capabilities": capabilities}


def payload_files(root):
    if not root.is_dir():
        raise PackageError("unavailable", "Package directory is unavailable", package=str(root))
    found = {}
    for current, dirs, names in os.walk(root, followlinks=False):
        current = Path(current)
        for name in list(dirs):
            child = current / name
            if child.is_symlink():
                raise PackageError("unsafe_payload", "Payload directories must not be symbolic links", path=str(child))
            if name == "__pycache__":
                dirs.remove(name)
        for name in names:
            child = current / name
            rel = child.relative_to(root).as_posix()
            if child.is_symlink():
                raise PackageError("unsafe_payload", "Payload files must not be symbolic links", path=str(child))
            if rel in RECORDS or name.endswith((".pyc", ".pyo")):
                continue
            relative(rel)
            if not child.is_file():
                raise PackageError("unsafe_payload", "Payload must contain regular files only", path=str(child))
            found[rel] = sha256(child.read_bytes())
    return dict(sorted(found.items()))


def load_manifest(root):
    if not root.is_dir():
        raise PackageError("unavailable", "Package directory is unavailable", package=str(root))
    path = root / MANIFEST
    if not path.exists():
        raise PackageError("unpublished", "Package has no release manifest", package=str(root))
    manifest = read_json(path)
    info = metadata(manifest)
    if set(manifest) != set(info) | {"files", "content_sha256"}:
        raise PackageError("invalid_manifest", "Release manifest has missing or unexpected fields")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise PackageError("invalid_manifest", "Release manifest requires a non-empty files mapping")
    for path, digest in files.items():
        relative(path)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise PackageError("invalid_manifest", "Invalid SHA256 payload digest", path=path)
    expected = sha256(canonical({**info, "files": files}))
    if manifest.get("content_sha256") != expected:
        raise PackageError("invalid_manifest", "Release metadata/content identity does not match its digest")
    required = {"README.md", "SKILL.md", "scripts/package.py"}
    required.update(capability["path"] for capability in info["capabilities"])
    required.update(str(PurePosixPath(capability["path"]).parent / "README.md") for capability in info["capabilities"])
    if not required <= files.keys():
        raise PackageError("invalid_manifest", "Manifest lacks required entry, tool or capability", missing=sorted(required - files.keys()))
    return manifest


def verified(root):
    manifest = load_manifest(root)
    actual = payload_files(root)
    expected = manifest["files"]
    differences = {"missing": sorted(expected.keys() - actual.keys()),
                   "extra": sorted(actual.keys() - expected.keys()),
                   "changed": sorted(path for path in actual.keys() & expected.keys() if actual[path] != expected[path])}
    if any(differences.values()):
        raise PackageError("source_modified", "Contents differ from the published manifest", differences=differences)
    return manifest


def manifest_digest(root):
    return sha256((root / MANIFEST).read_bytes())


def inspect(root):
    try:
        manifest = verified(root)
        return {"state": "valid", "manifest": manifest, "manifest_sha256": manifest_digest(root)}
    except (PackageError, OSError) as exc:
        if isinstance(exc, PackageError):
            return {"state": exc.state, "message": str(exc), **exc.details}
        return {"state": "unavailable", "message": str(exc)}


def origin(manifest):
    return manifest["name"], manifest["bindings"]


def record(root, filename=INSTALLATION):
    path = root / filename
    if not path.exists() and not path.is_symlink():
        return None
    return read_json(path, "customized")


def assert_record(root, manifest, item, filename=INSTALLATION):
    expected = {"schema_version": SCHEMA, "name": manifest["name"], "version": manifest["version"],
                "source_package": manifest["bindings"]["source_package"],
                "content_sha256": manifest["content_sha256"], "manifest_sha256": manifest_digest(root)}
    fields = {"created_at", "original_package"} if filename == SNAPSHOT else {"created_at", "scope", "skills_root", "project_root"}
    if not isinstance(item, dict) or set(item) != set(expected) | fields or any(item.get(key) != value for key, value in expected.items()):
        raise PackageError("customized", "Installation or snapshot identity or fields no longer match the package")


def identity_record(root, manifest, **extra):
    return {"schema_version": SCHEMA, "name": manifest["name"], "version": manifest["version"],
            "source_package": manifest["bindings"]["source_package"],
            "content_sha256": manifest["content_sha256"], "manifest_sha256": manifest_digest(root),
            "created_at": timestamp(), **extra}


def assert_installation_location(root, manifest, item):
    scope = item.get("scope")
    recorded = item.get("skills_root")
    if not isinstance(scope, str) or scope not in {"project", "shared"} or not isinstance(recorded, str) or str(absolute(recorded, "skills_root")) != recorded:
        raise PackageError("customized", "Installation record lacks a valid scope and canonical skills_root")
    if root.name != manifest["name"] or recorded != str(root.parent):
        raise PackageError("installation_mismatch", "Installation record belongs to another receiving Skill directory", package=str(root), recorded_skills_root=recorded, actual_skills_root=str(root.parent))
    project = item.get("project_root")
    if scope == "shared":
        if project is not None:
            raise PackageError("customized", "Shared installation must not claim a project_root")
    elif root.parent.name != "skills" or root.parent.parent.name != ".agents" or project != str(root.parent.parent.parent):
        raise PackageError("installation_mismatch", "Installation record does not match the actual receiving project", package=str(root), recorded_project=project, actual_project=str(root.parent.parent.parent))


def valid_for_use(root, check_project=True):
    manifest = verified(root)
    for filename in (INSTALLATION, SNAPSHOT):
        item = record(root, filename)
        if item is not None:
            assert_record(root, manifest, item, filename)
            if filename == INSTALLATION and check_project:
                assert_installation_location(root, manifest, item)
    return manifest


def status(root, source=None):
    local = inspect(root)
    result = {"package": str(root), "local_state": local["state"]}
    if local["state"] != "valid":
        return {**result, "state": "customized" if (root / INSTALLATION).exists() else local["state"],
                "details": {key: value for key, value in local.items() if key != "manifest"}}
    manifest = local["manifest"]
    result.update(version=manifest["version"], content_sha256=manifest["content_sha256"])
    try:
        installation = record(root)
        snapshot_record = record(root, SNAPSHOT)
        for filename, item in ((INSTALLATION, installation), (SNAPSHOT, snapshot_record)):
            if item is not None:
                assert_record(root, manifest, item, filename)
        if installation is not None:
            assert_installation_location(root, manifest, installation)
    except PackageError as exc:
        return {**result, "state": exc.state, "message": str(exc)}
    source = source or absolute(manifest["bindings"]["source_package"], "source_package")
    release = inspect(source)
    result.update(source=str(source), source_state=release["state"])
    if snapshot_record is not None:
        return {**result, "state": "fixed_snapshot"}
    if source == root:
        return {**result, "state": "source_valid"}
    if installation is None:
        return {**result, "state": "unrecorded"}
    if release["state"] != "valid":
        return {**result, "state": "installed_source_unavailable", "source_details": release}
    latest = release["manifest"]
    result["source_version"] = latest["version"]
    if origin(manifest) != origin(latest):
        return {**result, "state": "different_source"}
    if latest["version"] > manifest["version"]:
        return {**result, "state": "outdated"}
    if latest["version"] < manifest["version"]:
        return {**result, "state": "ahead"}
    if local["manifest_sha256"] != release["manifest_sha256"]:
        return {**result, "state": "release_conflict"}
    return {**result, "state": "current"}


def publish(root, version, metadata_path=None):
    if metadata_path is None and not (root / MANIFEST).exists():
        raise PackageError("unpublished", "Initial publication requires explicit --metadata")
    raw = read_json(metadata_path) if metadata_path else read_json(root / MANIFEST)
    info = metadata(raw, version)
    if Path(info["bindings"]["source_package"]) != root:
        raise PackageError("different_source", "Publishing is allowed only at the bound source_package")
    old = None
    if (root / MANIFEST).exists():
        old_raw = read_json(root / MANIFEST)
        if isinstance(old_raw, dict) and "files" in old_raw:
            old = load_manifest(root)
    files = payload_files(root)
    body = {**info, "files": files}
    candidate = {**body, "content_sha256": sha256(canonical(body))}
    required = {"README.md", "SKILL.md", "scripts/package.py"}
    required.update(capability["path"] for capability in info["capabilities"])
    required.update(str(PurePosixPath(capability["path"]).parent / "README.md") for capability in info["capabilities"])
    if not required <= files.keys():
        raise PackageError("incomplete_package", "Source lacks required complete-package materials", missing=sorted(required - files.keys()))
    if old and version <= old["version"]:
        if version == old["version"] and candidate == old:
            verified(root)
            return {"state": "already_published", "package": str(root), "version": version,
                    "content_sha256": candidate["content_sha256"]}
        raise PackageError("version_not_increasing", "Changed releases require a version greater than the previous release", previous_version=old["version"])
    write_json(root / MANIFEST, candidate)
    verified(root)
    return {"state": "published", "package": str(root), "version": version,
            "content_sha256": candidate["content_sha256"], "file_count": len(files)}


def is_within(path, parent):
    return path == parent or parent in path.parents


def check_destination(path, anchor):
    """Do not follow .agents or target symlinks into an unrelated worktree."""
    cursor = path
    while cursor != anchor:
        if cursor.is_symlink():
            raise PackageError("unsafe_target", "Installation/snapshot path must not contain symbolic links", path=str(cursor))
        if cursor.exists() and not cursor.is_dir():
            raise PackageError("invalid_target", "A destination directory component is a file", path=str(cursor))
        cursor = cursor.parent


def copy_payload(source, destination, manifest):
    for rel in manifest["files"]:
        item = source / rel
        if item.is_symlink():
            raise PackageError("unsafe_payload", "Source changed to a symbolic link during copy", path=str(item))
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
    shutil.copy2(source / MANIFEST, destination / MANIFEST)
    copied = verified(destination)
    if copied != manifest:
        raise PackageError("source_changed", "Source release changed while preparing a complete copy")


def install(root, project=None, source=None, allow_update=False, skills_dir=None, *, _source_manifest=None, _local_manifest=None):
    if source is None:
        source = absolute(load_manifest(root)["bindings"]["source_package"], "source_package")
    release = _source_manifest if _source_manifest is not None else verified(source)
    if Path(release["bindings"]["source_package"]) != source:
        raise PackageError("different_source", "Requested source is not the bound authoritative source_package")
    if (project is None) == (skills_dir is None):
        raise PackageError("invalid_target", "Choose exactly one receiving project or shared skills directory")
    name = release["name"]
    scope = "project" if project is not None else "shared"
    skills_root = project / ".agents" / "skills" if project is not None else skills_dir
    location = {"scope": scope, "skills_root": str(skills_root),
                "project_root": str(project) if project is not None else None}
    target = skills_root / name
    if is_within(target, source) or is_within(source, target):
        raise PackageError("invalid_target", "Source and installation directory must not overlap")
    check_destination(target, project.parent if project is not None else skills_root.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / f".{name}.install-lock"
    try:
        lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PackageError("installation_busy", "Another installation is using this target", path=str(lock)) from exc
    stage = None
    backup = None
    try:
        os.close(lock_fd)
        state = "installed"
        before = None
        if target.exists():
            local = ({"state": "valid", "manifest": _local_manifest,
                      "manifest_sha256": manifest_digest(target)}
                     if _local_manifest is not None else inspect(target))
            if local["state"] != "valid":
                raise PackageError("customized", "Existing target is not a pristine complete release; preserve it", details=local)
            installed = local["manifest"]
            if origin(installed) != origin(release):
                raise PackageError("different_source", "Existing same-name package belongs to another source")
            installed_record = record(target)
            project_mismatch = None
            if installed_record is not None:
                assert_record(target, installed, installed_record)
                try:
                    assert_installation_location(target, installed, installed_record)
                    if any(installed_record.get(key) != value for key, value in location.items()):
                        raise PackageError("installation_mismatch", "Installation record scope differs from the requested receiving scope")
                except PackageError as exc:
                    if exc.state != "installation_mismatch":
                        raise
                    project_mismatch = exc
            before = (installed, local["manifest_sha256"], json_bytes(installed_record) if installed_record else None)
            if installed == release and local["manifest_sha256"] == manifest_digest(source):
                refreshed = installed_record is None or project_mismatch is not None
                if refreshed:
                    write_json(target / INSTALLATION, identity_record(target, installed, **location))
                    valid_for_use(target)
                return {"state": "reused", "project": location["project_root"], "scope": scope, "skills_root": str(skills_root), "installation": str(target),
                        "version": installed["version"], "content_sha256": installed["content_sha256"],
                        "installation_record_refreshed": refreshed}
            if project_mismatch is not None:
                raise project_mismatch
            if installed_record is None:
                raise PackageError("unrecorded", "Existing different contents have no installation provenance; preserve them")
            if release["version"] <= installed["version"]:
                raise PackageError("release_conflict", "Refuse same-version replacement or downgrade")
            if not allow_update:
                raise PackageError("update_required", "A newer complete release exists; update permission is required", source_version=release["version"], installed_version=installed["version"])
            state = "updated"
        stage = Path(tempfile.mkdtemp(prefix=f".{name}-stage-", dir=target.parent))
        copy_payload(source, stage, release)
        write_json(stage / INSTALLATION, identity_record(stage, release, **location))
        assert_record(stage, release, record(stage))
        if before is not None:
            try:
                now = valid_for_use(target)
                current_record = record(target)
            except PackageError as exc:
                raise PackageError("target_changed", "Target changed while preparing the update; preserve it", cause=exc.state) from exc
            current = (now, manifest_digest(target), json_bytes(current_record) if current_record else None)
            if current != before:
                raise PackageError("target_changed", "Target changed while preparing the update; preserve it")
            backup = target.parent / f".{name}-backup-{uuid.uuid4().hex}"
            os.replace(target, backup)
        try:
            os.replace(stage, target)
            stage = None
        except OSError:
            if backup is not None:
                os.replace(backup, target)
                backup = None
            raise
        try:
            valid_for_use(target)
        except (PackageError, OSError) as exc:
            if backup is None:
                raise PackageError("installation_verification_failed", "Installed files could not be verified; retain the actual new directory for review", installation=str(target), restored_previous=False, cause=str(exc)) from exc
            failed = target.parent / f".{name}-failed-{uuid.uuid4().hex}"
            try:
                # Retain the failed new directory, including any concurrent edit.
                os.replace(target, failed)
                os.replace(backup, target)
                backup = None
            except OSError as rollback_error:
                raise PackageError("rollback_failed", "Final verification failed and previous installation could not be restored", installation=str(target), previous_installation=str(backup), retained_failed_package=str(failed), cause=str(exc), rollback_error=str(rollback_error)) from rollback_error
            raise PackageError("installation_verification_failed", "Final verification failed; previous complete installation restored and new directory retained", installation=str(target), restored_previous=True, retained_failed_package=str(failed), cause=str(exc)) from exc
        result = {"state": state, "project": location["project_root"], "scope": scope, "skills_root": str(skills_root), "installation": str(target),
                  "version": release["version"], "content_sha256": release["content_sha256"],
                  "file_count": len(release["files"])}
        if backup is not None:
            # Keep the old directory even after success: edits made after the
            # final target comparison must remain recoverable by the user.
            result["previous_installation"] = str(backup)
        return result
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        lock.unlink(missing_ok=True)


def snapshot(root, output, *, _manifest=None):
    manifest = _manifest if _manifest is not None else valid_for_use(root)
    if is_within(output, root) or is_within(root, output):
        raise PackageError("invalid_target", "Task snapshot and package directories must not overlap")
    if output.exists() or output.is_symlink():
        raise PackageError("snapshot_exists", "Task snapshot output must not exist", output=str(output))
    check_destination(output, Path(output.anchor))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-stage-", dir=output.parent))
    try:
        copy_payload(root, stage, manifest)
        write_json(stage / SNAPSHOT, identity_record(stage, manifest, original_package=str(root)))
        assert_record(stage, manifest, record(stage, SNAPSHOT), SNAPSHOT)
        if output.exists() or output.is_symlink():
            raise PackageError("snapshot_exists", "Task snapshot appeared during preparation; preserve it")
        # mkdir reserves the exact output without replacing any existing path.
        # Replacing this empty reservation publishes the already-complete copy.
        output.mkdir()
        try:
            os.replace(stage, output)
        except OSError:
            try:
                output.rmdir()
            except OSError:
                pass
            raise
        stage = None
        valid_for_use(output)
        return {"state": "snapshot_created", "snapshot": str(output), "version": manifest["version"],
                "content_sha256": manifest["content_sha256"], "file_count": len(manifest["files"])}
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def prepare(root, output, allow_update=False):
    """Verify one task version, update when authorized, and freeze its full package."""
    try:
        manifest = valid_for_use(root)
    except PackageError as exc:
        if (root / INSTALLATION).exists() and exc.state != "installation_mismatch":
            raise PackageError("customized", "Installed package is not intact; preserve local contents", cause=exc.state) from exc
        raise
    if record(root, SNAPSHOT) is not None:
        return {"state": "ready", "snapshot": str(root), "version": manifest["version"],
                "content_sha256": manifest["content_sha256"], "freshness": "fixed_snapshot",
                "source_checked": False, "updated": False, "limitations": []}

    source = absolute(manifest["bindings"]["source_package"], "source_package")
    freshness, source_checked, updated, limitations = "current", True, False, []
    previous_installation = None
    if source != root:
        installation = record(root)
        if installation is None:
            raise PackageError("unrecorded", "Mirror has no installation provenance; install it before preparing a task")
        release = inspect(source)
        if release["state"] != "valid":
            if release["state"] not in {"unavailable", "unpublished", "source_modified"}:
                raise PackageError(release["state"], "Source cannot be trusted for task preparation; preserve the installed package", source_details=release)
            unavailable = release["state"] == "unavailable"
            freshness = "source_unavailable" if unavailable else "source_unpublished"
            source_checked = not unavailable
            limitations.append("Source is unavailable; latest version was not confirmed." if unavailable else
                               "Source has no valid published release; used the intact installed version.")
        else:
            latest = release["manifest"]
            if origin(manifest) != origin(latest):
                raise PackageError("different_source", "Published source identity differs from this installed package")
            if latest["version"] < manifest["version"]:
                raise PackageError("ahead", "Installed version is ahead of the published source; refuse automatic replacement")
            if latest["version"] == manifest["version"]:
                if manifest_digest(root) != release["manifest_sha256"]:
                    raise PackageError("release_conflict", "Same-version source and mirror have different release identities")
            elif not allow_update:
                freshness = "outdated"
                limitations.append("A newer release exists; update was not authorized, so this task uses the intact installed version.")
            else:
                scope = installation["scope"]
                result = install(root, root.parent.parent.parent if scope == "project" else None,
                                 source, True, root.parent if scope == "shared" else None,
                                 _source_manifest=latest, _local_manifest=manifest)
                previous_installation = result.get("previous_installation")
                manifest, updated = latest, True

    try:
        frozen = snapshot(root, output, _manifest=manifest)
    except (PackageError, OSError) as exc:
        if not updated:
            raise
        details = {"updated": True, "version": manifest["version"], "installation": str(root),
                   "previous_installation": previous_installation}
        if isinstance(exc, PackageError):
            exc.details.update(details)
            raise
        raise PackageError("snapshot_failed", str(exc), **details) from exc
    result = {"state": "ready", "snapshot": frozen["snapshot"], "version": manifest["version"],
              "content_sha256": manifest["content_sha256"], "freshness": freshness,
              "source_checked": source_checked, "updated": updated, "limitations": limitations}
    if previous_installation is not None:
        result["previous_installation"] = previous_installation
    return result


def feedback(root, input_path):
    manifest = valid_for_use(root)
    value = read_json(input_path, "invalid_feedback")
    if not isinstance(value, dict):
        raise PackageError("invalid_feedback", "Feedback must be a JSON object")
    for field in ("task_id", "task_summary", "result_summary"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise PackageError("invalid_feedback", f"{field} must be a non-empty string")
    for field in ("capabilities_used", "knowledge_locations"):
        if not isinstance(value.get(field), list) or any(not isinstance(item, str) or not item.strip() for item in value[field]):
            raise PackageError("invalid_feedback", f"{field} must be a string list")
    used = value["capabilities_used"]
    names = {capability["name"] for capability in manifest["capabilities"]}
    if not used or len(used) != len(set(used)) or not set(used) <= names:
        raise PackageError("invalid_feedback", "capabilities_used must name distinct actual package capabilities")
    for location in value["knowledge_locations"]:
        if str(absolute(location, "knowledge_location")) != location:
            raise PackageError("invalid_feedback", "Knowledge locations must be canonical absolute paths", path=location)
    for field in ("limitations", "suggestions"):
        if field in value and (not isinstance(value[field], list) or any(not isinstance(item, str) for item in value[field])):
            raise PackageError("invalid_feedback", f"{field} must be a string list")
    folder = Path(manifest["bindings"]["feedback_root"])
    wiki = Path(manifest["bindings"]["wiki_root"])
    if not wiki.is_dir() or folder.is_symlink() or folder.resolve() != wiki / "query-feedback":
        raise PackageError("feedback_unavailable", "Wiki root is missing or feedback path is not the actual allowed directory", feedback_root=str(folder))
    if folder.exists() and not folder.is_dir():
        raise PackageError("feedback_unavailable", "Feedback path is not a directory", feedback_root=str(folder))
    folder.mkdir(exist_ok=True)
    query_id = uuid.uuid4().hex
    path = folder / f"{dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%dT%H%M%S')}-{query_id}.json"
    body = {**value, "query_id": query_id, "created_at": timestamp(), "timezone": "Asia/Taipei",
            "package": manifest["name"], "package_version": manifest["version"],
            "content_sha256": manifest["content_sha256"], "wiki_root": str(wiki)}
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json_bytes(body))
            handle.flush()
            os.fsync(handle.fileno())
        if read_json(path, "feedback_failed") != body:
            raise PackageError("feedback_failed", "Feedback contents could not be verified", feedback=str(path))
    except Exception:
        # Only this command's exclusively-created file may be removed on failure.
        path.unlink(missing_ok=True)
        raise
    return {"state": "feedback_written", "feedback": str(path), "query_id": query_id,
            "version": manifest["version"], "content_sha256": manifest["content_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE), help="absolute current package directory")
    commands = parser.add_subparsers(dest="operation", required=True)
    release = commands.add_parser("publish")
    release.add_argument("--version", type=int, required=True)
    release.add_argument("--metadata", help="initial business metadata JSON; later releases may reuse package.json")
    commands.add_parser("verify")
    inspect_cmd = commands.add_parser("status")
    inspect_cmd.add_argument("--source")
    install_cmd = commands.add_parser("install")
    receiving = install_cmd.add_mutually_exclusive_group(required=True)
    receiving.add_argument("--project")
    receiving.add_argument("--skills-dir", help="absolute shared Skill directory")
    install_cmd.add_argument("--source")
    install_cmd.add_argument("--allow-update", action="store_true")
    freeze = commands.add_parser("snapshot")
    freeze.add_argument("--output", required=True)
    ready = commands.add_parser("prepare")
    ready.add_argument("--output", required=True)
    ready.add_argument("--allow-update", action="store_true")
    report = commands.add_parser("feedback")
    report.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    try:
        root = absolute(args.package, "package")
        if args.operation == "publish":
            result = publish(root, args.version, absolute(args.metadata, "metadata") if args.metadata else None)
        elif args.operation == "verify":
            item = valid_for_use(root)
            result = {"state": "valid", "package": str(root), "version": item["version"],
                      "content_sha256": item["content_sha256"], "file_count": len(item["files"])}
        elif args.operation == "status":
            result = status(root, absolute(args.source, "source") if args.source else None)
        elif args.operation == "install":
            result = install(root, destination(args.project, "project") if args.project else None,
                             absolute(args.source, "source") if args.source else None, args.allow_update,
                             destination(args.skills_dir, "skills_dir") if args.skills_dir else None)
        elif args.operation == "snapshot":
            result = snapshot(root, destination(args.output, "output"))
        elif args.operation == "prepare":
            result = prepare(root, destination(args.output, "output"), args.allow_update)
        else:
            result = feedback(root, absolute(args.input, "input"))
        print(json.dumps({"ok": True, "operation": args.operation, **result}, ensure_ascii=False, sort_keys=True))
        return 0
    except PackageError as exc:
        print(json.dumps({"ok": False, "operation": args.operation, "state": exc.state,
                          "message": str(exc), **exc.details}, ensure_ascii=False, sort_keys=True))
        return 2
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "operation": args.operation, "state": "io_error",
                          "message": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
