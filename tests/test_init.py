from pathlib import Path

from strain_relief import project_dir, src_dir, test_dir


def test_dirs_exist():
    assert Path(project_dir).is_dir()
    assert Path(src_dir).is_dir()
    assert Path(test_dir).is_dir()
