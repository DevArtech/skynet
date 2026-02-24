from __future__ import annotations

from pathlib import Path


def checkpoint_iteration(path: Path) -> int:
    name = path.stem
    if "checkpoint_iter_" not in name:
        return -1
    try:
        return int(name.split("checkpoint_iter_")[-1])
    except ValueError:
        return -1


def find_latest_checkpoint(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("checkpoint_iter_*.pt"), key=checkpoint_iteration)
    if not candidates:
        return None
    return candidates[-1]
