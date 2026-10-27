"""Verify immutable vendor and reference snapshot hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from .paths import config_path, workspace_root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_entries(entries: list[dict], path_key: str) -> tuple[int, list[str]]:
    root = workspace_root()
    failures: list[str] = []
    for entry in entries:
        relative = entry[path_key]
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        else:
            actual = sha256(path)
            if actual != entry["sha256"]:
                failures.append(f"hash mismatch: {relative} expected={entry['sha256']} actual={actual}")
    return len(entries), failures


def verify_manifests() -> tuple[int, int]:
    vendor = json.loads(config_path("vendor_manifest.json").read_text(encoding="utf-8"))
    references = yaml.safe_load(config_path("reference_sources.yaml").read_text(encoding="utf-8"))
    vendor_count, failures = _verify_entries(vendor["files"], "copied_relative_path")
    reference_count, reference_failures = _verify_entries(references["references"], "copied_relative_path")
    failures.extend(reference_failures)
    if failures:
        raise RuntimeError("snapshot verification failed:\n" + "\n".join(failures))
    return vendor_count, reference_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    vendor_count, reference_count = verify_manifests()
    print(f"PASS: {vendor_count} vendor files and {reference_count} reference files match SHA256 manifests")


if __name__ == "__main__":
    main()
