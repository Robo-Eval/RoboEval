"""Per-episode logs, results.json aggregation, and provenance.

Shared by the evaluator (which writes these files) and the validator (which
recomputes them), so both sides always agree on what a metric means.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from roboeval.challenge import spec
from roboeval.challenge.determinism import math_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[2]

# Files whose contents define the benchmark. Editing any of them changes the
# source fingerprint recorded in results.json.
_FINGERPRINT_GLOBS = (
    "roboeval/**/*.py",
    "roboeval/**/*.xml",
    "roboeval/**/*.yaml",
    "roboeval/**/*.json",
    "thirdparty/mujoco_menagerie/franka_emika_panda/*.xml",
)


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and int dict keys into plain JSON types."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def episode_metrics(raw: Dict[str, Any], steps: int, control_frequency: int) -> Dict[str, float]:
    """Map the env's raw final metrics onto the 14 leaderboard keys."""
    out = {}
    for key, raw_key in spec.RAW_METRIC_KEYS.items():
        value = raw.get(raw_key)
        # Jerk needs >= 3 steps and is absent for very short episodes.
        out[key] = float(value) if isinstance(value, (int, float)) else 0.0
    out["tl"] = float(steps)
    out["ct"] = float(steps) / float(control_frequency)
    return {k: out[k] for k in spec.METRIC_KEYS}


def aggregate_variation(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean and population std of each metric over a variation's episodes."""
    metrics = {}
    for key in spec.METRIC_KEYS:
        values = np.array([ep["metrics"][key] for ep in episodes], dtype=np.float64)
        metrics[key] = {"mean": float(values.mean()), "std": float(values.std())}
    terminations = {t: 0 for t in spec.TERMINATIONS}
    for ep in episodes:
        terminations[ep["termination"]] += 1
    return {"episodes": len(episodes), "metrics": metrics, "terminations": terminations}


def summarize(results: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Per-family and suite means, every variation weighted equally."""

    def mean_over(entries: Iterable[Dict[str, Any]]) -> Dict[str, float]:
        entries = list(entries)
        return {
            k: float(np.mean([e["metrics"][k]["mean"] for e in entries])) for k in spec.METRIC_KEYS
        }

    per_family = {fam: mean_over(vs.values()) for fam, vs in results.items() if vs}
    every = [e for vs in results.values() for e in vs.values()]
    return {"per_family": per_family, "suite": mean_over(every) if every else {}}


def build_results(
    method_name: str,
    seeds: List[int],
    episodes_by_variation: Dict[tuple, List[Dict[str, Any]]],
    interface: Dict[str, Any],
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    results: Dict[str, Dict[str, Any]] = {}
    for family, variation in spec.all_variations():
        episodes = episodes_by_variation.get((family, variation))
        if episodes:
            results.setdefault(family, {})[variation] = aggregate_variation(episodes)
    return {
        "roboeval_release": spec.CHALLENGE_RELEASE,
        "harness_version": spec.HARNESS_VERSION,
        "method_name": method_name,
        "episodes_per_variation": len(seeds),
        "seeds": list(seeds),
        "interface": interface,
        "results": results,
        "summary": summarize(results),
        "provenance": provenance,
    }


def read_episode_logs(logs_dir: Path) -> Dict[tuple, List[Dict[str, Any]]]:
    """Load every logs/<Family>_<Variation>/episode_<i>.json, ordered by index."""
    out: Dict[tuple, List[Dict[str, Any]]] = {}
    for family, variation in spec.all_variations():
        vdir = Path(logs_dir) / spec.variation_dirname(family, variation)
        files = sorted(vdir.glob("episode_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        if files:
            out[(family, variation)] = [json.loads(p.read_text()) for p in files]
    return out


# ─────────────────────────────── provenance ──────────────────────────────
def source_fingerprint(root: Path = REPO_ROOT) -> str:
    """SHA-256 over the benchmark's source files (paths and contents)."""
    digest = hashlib.sha256()
    files = sorted({p for g in _FINGERPRINT_GLOBS for p in root.glob(g) if p.is_file()})
    for path in files:
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    try:
        import roboeval_metrics

        digest.update(Path(roboeval_metrics.__file__).with_name("evaluator.py").read_bytes())
    except (ImportError, OSError):
        pass
    return digest.hexdigest()


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_provenance(command: List[str]) -> Dict[str, Any]:
    import mujoco

    import roboeval

    status = _git("status", "--porcelain", "--untracked-files=no", "--", "roboeval", "thirdparty")
    try:
        from importlib.metadata import version

        metrics_version = version("roboeval-metrics")
    except Exception:
        metrics_version = None
    return {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "roboeval_version": getattr(roboeval, "__version__", None),
        "roboeval_metrics_version": metrics_version,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_describe": _git("describe", "--tags", "--always"),
        # Only tracked benchmark files count; untracked files are ignored.
        "git_dirty": bool(status) if status is not None else None,
        "source_fingerprint": source_fingerprint(),
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "math": math_fingerprint(),
        "command": " ".join(command),
    }


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()
