"""Run a submission's policy over the challenge suite.

Writes, inside the submission directory (or ``--output-dir``):

    logs/<Family>_<Variation>/episode_<i>.json   per-episode metrics and outcome
    logs/<Family>_<Variation>/episode_<i>.npz    the exact actions sent to the env
    results.json                                 aggregated from logs/

Usage:
    python -m roboeval.challenge.evaluate --submission submission/ --workers 8 --gpus 0,1,2,3

    # quick smoke test: 2 episodes of one family
    python -m roboeval.challenge.evaluate --submission submission/ --tasks LiftPot --episodes 2
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboeval.challenge import results as res
from roboeval.challenge import spec
from roboeval.challenge.determinism import ensure_pinned
from roboeval.challenge.submission import (
    EpisodeSpec,
    Interface,
    load_metadata,
    load_policy,
    make_env,
    task_instruction,
)

logger = logging.getLogger("roboeval.challenge")


def _seed_policy_rngs(policy_seed: int, env_seed: int) -> None:
    """Seed Python and (if loaded) torch RNGs; env.reset already seeded NumPy."""
    seed = (policy_seed * 1_000_003 + env_seed) % (2**32)
    random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)


def run_episode(env, policy, episode: EpisodeSpec) -> Dict[str, Any]:
    """Roll out one episode. Never raises: errors become a failed episode."""
    actions: List[np.ndarray] = []
    info: Dict[str, Any] = {}
    termination, error = "budget", None
    start = time.time()
    try:
        obs, info = env.reset(seed=episode.seed)
        _seed_policy_rngs(episode.policy_seed, episode.seed)
        policy.reset(episode)
        for _ in range(episode.max_steps):
            action = np.asarray(policy.act(obs), dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            actions.append(action)
            if env.success:
                termination = "success"
                break
            if terminated:
                termination = "fail"
                break
            if truncated:
                termination = "unhealthy"
                break
    except Exception:
        termination, error = "error", traceback.format_exc(limit=8)
    steps = len(actions)
    raw = res.to_jsonable(info)
    if termination == "error":
        raw["success"] = 0.0
    return {
        "family": episode.family,
        "variation": episode.variation,
        "env_name": episode.env_name,
        "episode_index": episode.episode_index,
        "seed": episode.seed,
        "policy_seed": episode.policy_seed,
        "control_frequency": episode.control_frequency,
        "max_steps": episode.max_steps,
        "steps": steps,
        "termination": termination,
        "error": error,
        "sim_time_s": steps / episode.control_frequency,
        "wall_time_s": time.time() - start,
        "raw": raw,
        "metrics": res.episode_metrics(raw, steps, episode.control_frequency),
        "_actions": np.stack(actions) if actions else np.zeros((0,), np.float32),
    }


def write_episode(vdir: Path, record: Dict[str, Any]) -> None:
    i = record["episode_index"]
    actions = record.pop("_actions")
    npz_path = vdir / f"episode_{i}.npz"
    np.savez_compressed(npz_path, actions=actions)
    record["actions_file"] = npz_path.name
    record["actions_sha256"] = res.file_sha256(npz_path)
    tmp = vdir / f"episode_{i}.json.tmp"
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(vdir / f"episode_{i}.json")  # atomic: --resume never sees half a log


def evaluate_variation(job: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entry point: evaluate every seed of one task variation."""
    if job.get("gpu") is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(job["gpu"])
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(job["gpu"]))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    family, variation = job["family"], job["variation"]
    submission_dir, vdir = Path(job["submission_dir"]), Path(job["variation_dir"])
    vdir.mkdir(parents=True, exist_ok=True)
    interface = Interface.from_dict(job["interface"])
    metadata = load_metadata(submission_dir)

    pending = [
        (i, s) for i, s in enumerate(job["seeds"])
        if not (vdir / f"episode_{i}.json").exists()
    ]
    if not pending:
        return {"family": family, "variation": variation, "ran": 0}

    policy = load_policy(submission_dir, metadata)
    env = make_env(family, variation, interface)
    max_steps = spec.max_steps(family, interface.control_frequency)
    instruction = task_instruction(family)
    start = time.time()
    for i, seed in pending:
        episode = EpisodeSpec(
            family=family,
            variation=variation,
            env_name=type(env).__name__,
            instruction=instruction,
            episode_index=i,
            seed=int(seed),
            policy_seed=int(metadata.get("policy_seed", 0)),
            control_frequency=interface.control_frequency,
            max_steps=max_steps,
            action_space=env.action_space,
        )
        record = run_episode(env, policy, episode)
        write_episode(vdir, record)
        if record["termination"] == "error":
            logger.warning("%s/%s episode %d errored:\n%s", family, variation, i, record["error"])
            env.close()
            env = make_env(family, variation, interface)
    env.close()
    logger.info("%s/%s: %d episodes in %.0fs", family, variation, len(pending), time.time() - start)
    return {"family": family, "variation": variation, "ran": len(pending)}


def select_variations(filters: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    """Resolve ``--tasks LiftPot LiftTray/PositionAndOrientation`` against the official list."""
    every = spec.all_variations()
    if not filters:
        return every
    chosen = []
    for f in filters:
        family, _, variation = f.partition("/")
        matches = [(fa, va) for fa, va in every if fa == family and (not variation or va == variation)]
        if not matches:
            valid = ", ".join(f"{fa}/{va}" for fa, va in every)
            raise SystemExit(f"--tasks: '{f}' matches nothing. Valid entries: {valid}")
        chosen += [m for m in matches if m not in chosen]
    return chosen


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a challenge submission and write logs/ and results.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--submission", required=True, type=Path,
                        help="Submission directory containing metadata.yaml, policy/ and checkpoint/.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write logs/ and results.json (default: the submission directory).")
    parser.add_argument("--tasks", nargs="+", default=None, metavar="FAMILY[/VARIATION]",
                        help="Evaluate a subset, e.g. LiftPot or LiftTray. The result is partial "
                             "and will not pass validation.")
    parser.add_argument("--episodes", type=int, default=None,
                        help=f"Use only the first N official seeds (default: all {len(spec.EVAL_SEEDS)}).")
    parser.add_argument("--seeds-file", type=Path, default=None,
                        help="JSON list of seeds to use instead of the official list (organizer re-runs).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel processes; each evaluates one variation at a time.")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU ids assigned round-robin to workers for EGL rendering "
                             "and CUDA, e.g. 0,1,2,3.")
    parser.add_argument("--resume", action="store_true",
                        help="Keep existing episode logs and run only the missing episodes.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing logs for the selected variations before running.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    submission_dir = args.submission.resolve()
    out_dir = (args.output_dir or submission_dir).resolve()
    logs_dir = out_dir / "logs"
    metadata = load_metadata(submission_dir)
    interface = Interface.from_dict(metadata["interface"])
    if interface.problems():
        for p in interface.problems():
            logger.error("metadata.yaml interface: %s", p)
        return 2
    if interface.cameras and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    if args.seeds_file:
        seeds = [int(s) for s in json.loads(args.seeds_file.read_text())]
    else:
        seeds = list(spec.EVAL_SEEDS)
    if args.episodes is not None:
        seeds = seeds[: args.episodes]

    variations = select_variations(args.tasks)
    for family, variation in variations:
        vdir = logs_dir / spec.variation_dirname(family, variation)
        if vdir.exists() and any(vdir.iterdir()):
            if args.overwrite:
                shutil.rmtree(vdir)
            elif not args.resume:
                logger.error("%s already has logs. Pass --resume to continue or --overwrite to restart.", vdir)
                return 2
        # Logs from a different seed list must never be mixed into one run.
        for p in sorted(vdir.glob("episode_*.json")) if vdir.exists() else []:
            i = int(p.stem.split("_")[1])
            if i >= len(seeds) or json.loads(p.read_text())["seed"] != seeds[i]:
                logger.error("%s was produced with a different seed list; use --overwrite.", p)
                return 2

    gpus = [g.strip() for g in args.gpus.split(",")] if args.gpus else []
    if interface.cameras and args.workers > 1 and not gpus:
        logger.warning("All %d workers will render on GPU 0; pass --gpus to spread them.", args.workers)
    jobs = [
        {
            "family": family,
            "variation": variation,
            "seeds": seeds,
            "submission_dir": str(submission_dir),
            "variation_dir": str(logs_dir / spec.variation_dirname(family, variation)),
            "interface": interface.to_dict(),
            "gpu": gpus[k % len(gpus)] if gpus else None,
        }
        for k, (family, variation) in enumerate(variations)
    ]
    logger.info("Evaluating %d variations x %d episodes with %d worker(s)", len(jobs), len(seeds), args.workers)
    start = time.time()
    if args.workers <= 1:
        for job in jobs:
            evaluate_variation(job)
    else:
        # One process per variation: each gets a fresh MuJoCo/EGL context on its GPU.
        with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
            for _ in pool.imap_unordered(evaluate_variation, jobs):
                pass

    episodes = res.read_episode_logs(logs_dir)
    episodes = {k: v[: len(seeds)] for k, v in episodes.items()}
    results = res.build_results(
        method_name=metadata.get("method_name", ""),
        seeds=seeds,
        episodes_by_variation=episodes,
        interface=interface.to_dict(),
        provenance=res.collect_provenance(sys.argv),
    )
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    logger.info("Done in %.0fs. Wrote %s", time.time() - start, out_dir / "results.json")
    print(f"\n{'variation':48s} {'eps':>4s} {'success':>8s} {'tp':>6s}  errors")
    for family, vs in results["results"].items():
        for variation, r in vs.items():
            m = r["metrics"]
            print(f"{family + '/' + variation:48s} {r['episodes']:4d} {m['success']['mean']:8.3f} "
                  f"{m['tp']['mean']:6.3f}  {r['terminations']['error']}")
    missing = [f"{f}/{v}" for f, v in spec.all_variations() if len(episodes.get((f, v), [])) < len(seeds)]
    if missing or seeds != list(spec.EVAL_SEEDS):
        print(f"\nThis is a partial run ({len(missing)} variation(s) incomplete or non-official seeds); "
              "it will not pass `python -m roboeval.challenge.validate`.")
    if any(r["terminations"]["error"] for vs in results["results"].values() for r in vs.values()):
        print("Some episodes raised exceptions; see the 'error' field in their logs.")
    return 0


if __name__ == "__main__":
    ensure_pinned("roboeval.challenge.evaluate", sys.argv[1:])
    sys.exit(main())
