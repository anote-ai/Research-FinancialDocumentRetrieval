"""Helpers for keeping each experiment invocation in its own results folder."""

from datetime import datetime
from pathlib import Path


def create_run_dir(condition: str, root: str = "results/runs") -> Path:
    """Create and return a unique output directory for one experiment run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(root) / condition / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Run outputs will be saved to {run_dir}")
    return run_dir
