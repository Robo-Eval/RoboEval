"""Human reference values for the challenge score.

The references describe the episodes of the training dataset,
roboeval_ee_delta_20hz_scaled, for the evaluated variant of each task. Each
episode's end-effector delta actions are replayed from its source demo's seed
at 20 Hz under the challenge rules (same step budget, stop at success); the
episodes that succeed give the median of each leaderboard metric.

    python -m roboeval.challenge.make_human_reference <dataset_root> [out_dir]

<dataset_root> holds one LeRobot dataset per variation, as downloaded with
`hf download helen9975/roboeval_ee_delta_20hz_scaled --repo-type dataset
--include "*PositionAndOrientation/data/*" "*PositionAndOrientation/meta/*"`.
Writes human_reference.json (used by scoring.py) and the per-episode
human_reference_episodes.json.
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

VARIATION = "PositionAndOrientation"


def load_episodes(variation_dir: Path):
    """[(uuid, actions [T, 14], first state [36])] in episode order, actions/uuid/state columns only."""
    import pyarrow.parquet as pq

    cols = ["episode_index", "frame_index", "uuid", "actions", "state"]
    tables = [pq.read_table(p, columns=cols) for p in sorted((variation_dir / "data").rglob("*.parquet"))]
    rows = {c: np.concatenate([np.asarray(t.column(c).to_pylist(), dtype=object) for t in tables]) for c in cols}
    episodes = []
    for ep in sorted(set(rows["episode_index"].tolist())):
        idx = np.flatnonzero(rows["episode_index"] == ep)
        idx = idx[np.argsort(rows["frame_index"][idx].astype(int))]
        episodes.append((str(rows["uuid"][idx[0]]),
                         np.stack([np.asarray(a, dtype=np.float32) for a in rows["actions"][idx]]),
                         np.asarray(rows["state"][idx[0]], dtype=np.float32)))
    return episodes


def demo_seeds(family_variant: str) -> dict:
    """uuid -> reset seed, read from the raw demos' metadata (no replay)."""
    from safetensors import safe_open

    from roboeval.demonstrations.demo_store import DemoStore

    root = DemoStore()._cache_path
    seeds = {}
    for p in root.rglob(f"{family_variant}/*/lightweight/*.safetensors"):
        with safe_open(str(p), "np") as f:
            meta = f.metadata()
        seeds[json.loads(meta["uuid"])] = int(meta["seed"])
    return seeds


def run_family(args):
    family, dataset_root = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    from roboeval.challenge import results as res
    from roboeval.challenge import spec
    from roboeval.challenge.submission import Interface, make_env

    episodes = load_episodes(Path(dataset_root) / f"{family}{VARIATION}")
    seeds = demo_seeds(f"{family}{VARIATION}")
    env = make_env(family, VARIATION, Interface(absolute=False, ee=True))
    max_steps = spec.max_steps(family, spec.CONTROL_FREQUENCY)
    out = []
    for uuid, actions, state0 in episodes:
        obs, info = env.reset(seed=seeds[uuid])
        start_err = float(np.abs(obs["proprioception"] - state0).max())
        success, steps = False, 0
        for a in actions[:max_steps]:
            _, _, terminated, truncated, info = env.step(np.clip(a, env.action_space.low, env.action_space.high))
            steps += 1
            if env.success:
                success = True
                break
            if terminated or truncated:
                break
        out.append({"uuid": uuid, "seed": seeds[uuid], "success": success, "steps": steps,
                    "dataset_steps": int(len(actions)), "start_state_error": start_err,
                    "metrics": res.episode_metrics(res.to_jsonable(info), steps, spec.CONTROL_FREQUENCY)})
    env.close()
    ok = [e for e in out if e["success"]]
    median = {k: float(np.median([e["metrics"][k] for e in ok])) for k in spec.METRIC_KEYS} if ok else {}
    return family, {"dataset": "roboeval_ee_delta_20hz_scaled", "episodes": len(out), "successful": len(ok),
                    "median": median}, out


if __name__ == "__main__":
    from roboeval.challenge import spec
    from roboeval.demonstrations.demo_store import DemoStore

    dataset_root = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent
    DemoStore().pull_demos()  # seeds come from the raw demos; download once, before the workers start
    families = [f.key for f in spec.TASK_FAMILIES]
    with mp.get_context("spawn").Pool(len(families)) as pool:
        results = pool.map(run_family, [(f, str(dataset_root)) for f in families])
    (out_dir / "human_reference.json").write_text(json.dumps({f: s for f, s, _ in results}, indent=2))
    (out_dir / "human_reference_episodes.json").write_text(json.dumps({f: eps for f, _, eps in results}))
    for fam, s, eps in results:
        err = max((e["start_state_error"] for e in eps), default=0.0)
        print(f"{fam:24s} {s['successful']:3d}/{s['episodes']:3d} episodes succeed  (max start-state error {err:.2e})",
              flush=True)
