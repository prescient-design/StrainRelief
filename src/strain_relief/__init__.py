import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Directories
project_dir: Path = Path(__file__).resolve().parents[2]
src_dir: Path = project_dir / "src"
test_dir: Path = project_dir / "tests"
data_dir: Path = project_dir / "data"
