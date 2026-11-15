"""Resolve bundled input documents independently of the process working directory."""
from pathlib import Path

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"


def bundled_document(filename: str) -> Path:
    """Return an absolute path to one of the project's bundled input files."""
    if not filename or "\\" in filename or Path(filename).name != filename:
        raise ValueError("Expected a simple document filename, not a directory path")
    return DATA_DIRECTORY / filename
