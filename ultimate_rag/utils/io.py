"""I/O utility functions."""

import json
import pickle
import hashlib
import re
from pathlib import Path
from typing import Any, Optional
from loguru import logger


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path, indent: int = 2):
    """Save data to JSON file."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_pickle(path: Path) -> Any:
    """Load pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(data: Any, path: Path):
    """Save data to pickle file."""
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(path: Path, algorithm: str = "md5") -> str:
    """Get hash of file contents."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_filename(name: str, max_length: int = 100) -> str:
    """Convert string to safe filename."""
    safe = re.sub(r'[^\w\s-]', '', name)
    safe = re.sub(r'[-\s]+', '_', safe)
    return safe[:max_length].strip('_')


def read_text_file(path: Path, encoding: str = "utf-8") -> str:
    """Read text file."""
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write_text_file(content: str, path: Path, encoding: str = "utf-8"):
    """Write text file."""
    ensure_dir(path.parent)
    with open(path, "w", encoding=encoding) as f:
        f.write(content)


def list_files(
    directory: Path,
    pattern: str = "*",
    recursive: bool = False
) -> list[Path]:
    """List files matching pattern."""
    directory = Path(directory)
    if recursive:
        return list(directory.rglob(pattern))
    return list(directory.glob(pattern))


def get_checkpoint_path(base_dir: Path, name: str, stage: str) -> Path:
    """Get checkpoint file path."""
    return base_dir / "checkpoints" / f"{name}_{stage}.pkl"


def save_checkpoint(data: Any, base_dir: Path, name: str, stage: str):
    """Save processing checkpoint."""
    path = get_checkpoint_path(base_dir, name, stage)
    ensure_dir(path.parent)
    save_pickle(data, path)
    logger.debug(f"Saved checkpoint: {path}")


def load_checkpoint(base_dir: Path, name: str, stage: str) -> Optional[Any]:
    """Load processing checkpoint if exists."""
    path = get_checkpoint_path(base_dir, name, stage)
    if path.exists():
        logger.debug(f"Loading checkpoint: {path}")
        return load_pickle(path)
    return None
