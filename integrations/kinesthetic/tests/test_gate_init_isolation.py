"""A14's dataset-isolation claim must be falsifiable.

The claim is that the initialization chain never records an episode. The original check
watched an `EpisodeRecorder` catalog while `run_initialization(execute=False)` ran, but
`run_initialization` is never handed that recorder, so the catalog was empty by
construction: the check reduced to `before_run["success"]` and would have passed even if the
chain had grown a recording path. These tests pin the replacement, which reads the package
source, and prove the scanner actually rejects a module that can record.
"""

from pathlib import Path

from openarm_kinesthetic.validation.gate_init import (
    INITIALIZATION_PACKAGE_DIRECTORY,
    RECORDING_MARKERS,
    recording_capable_module_names,
)


def test_initialization_package_cannot_record_an_episode() -> None:
    assert INITIALIZATION_PACKAGE_DIRECTORY.is_dir()
    assert sorted(
        path.name for path in INITIALIZATION_PACKAGE_DIRECTORY.glob("*.py")
    ), "the scanner found no modules to scan, so A14 would pass vacuously"
    assert recording_capable_module_names(INITIALIZATION_PACKAGE_DIRECTORY) == []


def test_the_scanner_rejects_a_module_that_can_record(tmp_path: Path) -> None:
    """The scanner must fail on the thing it exists to catch."""
    (tmp_path / "clean.py").write_text("import math\n", encoding="utf-8")
    (tmp_path / "recording.py").write_text(
        "from ..recording.transaction import EpisodeRecorder\n\nr = EpisodeRecorder('/tmp')\n",
        encoding="utf-8",
    )
    assert RECORDING_MARKERS, "an empty marker list would make the scan meaningless"
    assert recording_capable_module_names(tmp_path) == ["recording.py"]


def test_the_scanner_includes_new_subpackages(tmp_path: Path) -> None:
    nested = tmp_path / "future_backend"
    nested.mkdir()
    (nested / "writer.py").write_text(
        "from somewhere import EpisodeRecorder\n", encoding="utf-8"
    )

    assert recording_capable_module_names(tmp_path) == ["future_backend/writer.py"]
