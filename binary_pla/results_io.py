"""
Save and load experiment results as JSON.

Each file is wrapped in a metadata envelope (notebook id, trial, timestamp,
git hash) and stored at results/{notebook_id}_{trial}.json.
"""

import json
import os
import subprocess
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "results")


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _result_path(notebook_id: str, trial: str, results_dir: str) -> str:
    return os.path.join(results_dir, f"{notebook_id}_{trial}.json")


def save_results(
    notebook_id:  str,
    trial:        str,
    results:      dict,
    results_dir:  str  = RESULTS_DIR,
    extra_meta:   dict = None,
) -> str:
    """Write results to results_dir/{notebook_id}_{trial}.json. Returns the path."""
    os.makedirs(results_dir, exist_ok=True)

    meta = {
        "notebook":  notebook_id,
        "trial":     trial,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash":  _git_hash(),
    }
    if extra_meta:
        meta.update(extra_meta)

    payload = {"meta": meta, "results": results}
    path = _result_path(notebook_id, trial, results_dir)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[results_io] saved → {path}")
    return path


def load_results(
    notebook_id: str,
    trial:       str,
    results_dir: str = RESULTS_DIR,
) -> tuple:
    """Load results. Returns (meta, results) tuple."""
    path = _result_path(notebook_id, trial, results_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results file not found: {path}\n"
            f"Run notebook '{notebook_id}' first to generate it."
        )

    with open(path) as f:
        payload = json.load(f)

    meta    = payload["meta"]
    results = payload["results"]
    print(f"[results_io] loaded {notebook_id}/{trial}  "
          f"(saved {meta.get('timestamp', '?')[:10]}, "
          f"git {meta.get('git_hash', '?')})")
    return meta, results


def results_exist(
    notebook_id: str,
    trial:       str,
    results_dir: str = RESULTS_DIR,
) -> bool:
    """Return True if the results file already exists."""
    return os.path.exists(_result_path(notebook_id, trial, results_dir))
