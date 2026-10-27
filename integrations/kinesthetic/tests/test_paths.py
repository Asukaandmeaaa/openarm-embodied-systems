from pathlib import Path

from openarm_kinesthetic.paths import config_path, vendor_path, workspace_root


def test_paths_do_not_depend_on_cwd(monkeypatch, tmp_path: Path) -> None:
    expected_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    assert workspace_root() == expected_root
    assert config_path("poses.yaml").is_file()
    assert vendor_path("v2", "openarm_bimanual.xml").is_file()
