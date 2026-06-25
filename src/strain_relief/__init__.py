from pathlib import Path

# Directories
project_dir: Path = Path(__file__).resolve().parents[2]
src_dir: Path = project_dir / "src"
test_dir: Path = project_dir / "tests"
data_dir: Path = project_dir / "data"

__all__ = [
    "compute_strain",
    "project_dir",
    "src_dir",
    "test_dir",
    "data_dir",
]


def __getattr__(name: str):
    # Lazy import: omegaconf pins antlr 4.9 which conflicts with the
    # antlr 4.13 that Dagster's asset-selection grammar needs.
    if name == "compute_strain":
        from strain_relief.compute_strain import compute_strain

        return compute_strain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
