from openarm_kinesthetic.manifest import _verify_entries, verify_manifests


def test_snapshot_manifests_match() -> None:
    vendor_count, reference_count = verify_manifests()
    assert vendor_count >= 1
    assert reference_count >= 1


def test_manifest_mismatch_is_explicit() -> None:
    entries = [{"copied_relative_path": "README.md", "sha256": "0" * 64}]
    count, failures = _verify_entries(entries, "copied_relative_path")
    assert count == 1
    assert failures and failures[0].startswith("hash mismatch:")
