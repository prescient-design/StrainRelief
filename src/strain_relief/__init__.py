from pathlib import Path

from strain_relief.compute_strain import compute_strain

# Directories
project_dir: Path = Path(__file__).resolve().parents[2]
src_dir: Path = project_dir / "src"
test_dir: Path = project_dir / "tests"
data_dir: Path = project_dir / "data"
config_dir: Path = src_dir / "strain_relief/hydra_config"

__all__ = [
    "compute_strain",
    "project_dir",
    "src_dir",
    "test_dir",
    "data_dir",
]
