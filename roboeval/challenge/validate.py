"""Check a submission bundle before uploading it.

Runs the same automated checks the organizers run on receipt:

  1. layout      required files and directories are present
  2. metadata    metadata.yaml has every field and a valid interface
  3. schema      results.json covers all variations, metrics, and official seeds
  4. consistency results.json is exactly re-derived from logs/
  5. provenance  release, fingerprint, and clean source tree
  6. replay      (--replay N) N logged episodes replayed in the simulator
                 reproduce their logged outcome and metrics

Usage:
    python -m roboeval.challenge.validate submission/
    python -m roboeval.challenge.validate submission/ --replay 30
Exit status is 0 only if no check FAILs.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from roboeval.challenge import results as res
from roboeval.challenge import spec
from roboeval.challenge.determinism import ensure_pinned, is_pinned, math_fingerprint
from roboeval.challenge.submission import (
    REQUIRED_METADATA_FIELDS,
    REQUIRED_PATHS,
    Interface,
    load_metadata,
)

# Re-derived aggregates must match to float round-off.
AGGREGATE_RTOL = 1e-9
# Replayed episodes must reproduce the logged metrics to this tolerance.
REPLAY_RTOL = 1e-4
REPLAY_ATOL = 1e-6


class Report:
    def __init__(self) -> None:
        self.items: List[Dict[str, str]] = []

    def add(self, level: str, check: str, message: str) -> None:
        self.items.append({"level": level, "check": check, "message": message})

    def fail(self, check: str, message: str) -> None:
        self.add("FAIL", check, message)

    def warn(self, check: str, message: str) -> None:
        self.add("WARN", check, message)

    def ok(self, check: str, message: str) -> None:
        self.add("PASS", check, message)

    @property
    def failed(self) -> bool:
        return any(i["level"] == "FAIL" for i in self.items)

    def print(self) -> None:
        for i in self.items:
            print(f"[{i['level']}] {i['check']:12s} {i['message']}")
        n = {lvl: sum(i["level"] == lvl for i in self.items) for lvl in ("PASS", "WARN", "FAIL")}
        print(f"\n{n['PASS']} passed, {n['WARN']} warnings, {n['FAIL']} failed")
        print("Submission is VALID." if not self.failed else "Submission is NOT valid; fix the FAIL items.")


def _close(a: float, b: float, rtol: float, atol: float = 0.0) -> bool:
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def check_layout(sub: Path, report: Report) -> None:
    missing = [p for p in REQUIRED_PATHS if not (sub / p).exists()]
    if missing:
        report.fail("layout", "missing: " + ", ".join(missing))
    else:
        report.ok("layout", "all required files and directories present")
    pdf = sub / "report.pdf"
    if pdf.exists():
        with open(pdf, "rb") as f:
            if f.read(5) != b"%PDF-":
                report.fail("layout", "report.pdf is not a PDF file")
    ckpt = sub / "checkpoint"
    if ckpt.is_dir() and not any(p.is_file() and p.name != "README.md" for p in ckpt.rglob("*")):
        report.warn("layout", "checkpoint/ holds no weights; checkpoint_url in metadata.yaml must point to them")


def check_metadata(sub: Path, report: Report) -> Optional[Dict[str, Any]]:
    before = sum(i["level"] == "FAIL" for i in report.items)
    try:
        meta = load_metadata(sub)
    except Exception as e:  # noqa: BLE001 - surface any parse error to the user
        report.fail("metadata", f"cannot read metadata.yaml: {e}")
        return None
    missing = [k for k in REQUIRED_METADATA_FIELDS if k not in meta]
    if missing:
        report.fail("metadata", "missing fields: " + ", ".join(missing))
    if meta.get("roboeval_release") != spec.CHALLENGE_RELEASE:
        report.fail("metadata", f"roboeval_release must be '{spec.CHALLENGE_RELEASE}'")
    if meta.get("team_name") in (None, "", "Your Team") or meta.get("method_name") in (None, "", "YourMethod-v2"):
        report.fail("metadata", "team_name/method_name still hold template placeholders")
    if "@" not in str(meta.get("contact_email", "")) or str(meta.get("contact_email")).endswith("example.edu"):
        report.fail("metadata", "contact_email is missing or a placeholder")
    if meta.get("open_source") and not meta.get("code_url"):
        report.warn("metadata", "open_source is true but code_url is empty")
    try:
        interface = Interface.from_dict(meta["interface"])
        for p in interface.problems():
            report.fail("metadata", f"interface: {p}")
    except (KeyError, TypeError, ValueError) as e:
        report.fail("metadata", f"interface block is malformed: {e!r}")
    if sum(i["level"] == "FAIL" for i in report.items) == before:
        report.ok("metadata", f"team '{meta['team_name']}', method '{meta['method_name']}'")
    return meta


def check_schema(results: Dict[str, Any], meta: Optional[Dict[str, Any]], report: Report) -> None:
    before = len([i for i in report.items if i["level"] == "FAIL"])
    if results.get("roboeval_release") != spec.CHALLENGE_RELEASE:
        report.fail("schema", f"roboeval_release is '{results.get('roboeval_release')}'")
    if meta and results.get("method_name") != meta.get("method_name"):
        report.fail("schema", "method_name differs between results.json and metadata.yaml")
    if meta and results.get("interface") != Interface.from_dict(meta["interface"]).to_dict():
        report.fail("schema", "interface in results.json differs from metadata.yaml; re-run the evaluator")
    if list(results.get("seeds", [])) != list(spec.EVAL_SEEDS):
        report.fail("schema", f"seeds are not the official {len(spec.EVAL_SEEDS)}-seed list "
                              "(was --episodes or --seeds-file used?)")
    table = results.get("results", {})
    missing = [f"{f}/{v}" for f, v in spec.all_variations() if v not in table.get(f, {})]
    if missing:
        report.fail("schema", f"{len(missing)} variation(s) missing: " + ", ".join(missing))
    extra = [f"{f}/{v}" for f, vs in table.items() for v in vs
             if f not in spec.FAMILIES_BY_KEY or v not in spec.FAMILIES_BY_KEY[f].variations]
    if extra:
        report.fail("schema", "unknown variation(s): " + ", ".join(extra))
    for f, v in spec.all_variations():
        entry = table.get(f, {}).get(v)
        if entry is None:
            continue
        if entry.get("episodes") != len(spec.EVAL_SEEDS):
            report.fail("schema", f"{f}/{v}: {entry.get('episodes')} episodes, expected {len(spec.EVAL_SEEDS)}")
        for k in spec.METRIC_KEYS:
            m = entry.get("metrics", {}).get(k)
            if not (isinstance(m, dict) and all(isinstance(m.get(s), (int, float)) for s in ("mean", "std"))):
                report.fail("schema", f"{f}/{v}: metric '{k}' missing or not numeric")
    if len([i for i in report.items if i["level"] == "FAIL"]) == before:
        report.ok("schema", f"{len(spec.all_variations())} variations x {len(spec.METRIC_KEYS)} metrics present")


def check_consistency(sub: Path, results: Dict[str, Any], report: Report) -> Dict[tuple, List[Dict[str, Any]]]:
    episodes = res.read_episode_logs(sub / "logs")
    seeds = list(results.get("seeds", []))
    problems = 0
    for (f, v), eps in episodes.items():
        if [e["seed"] for e in eps] != seeds[: len(eps)] or len(eps) != len(seeds):
            report.fail("consistency", f"{f}/{v}: logged seeds do not match results.json seeds")
            problems += 1
            continue
        for e in eps:
            vdir = sub / "logs" / spec.variation_dirname(f, v)
            npz = vdir / e.get("actions_file", "")
            if not npz.is_file() or res.file_sha256(npz) != e.get("actions_sha256"):
                report.fail("consistency", f"{f}/{v} episode {e['episode_index']}: actions file missing or altered")
                problems += 1
            derived = res.episode_metrics(e["raw"], e["steps"], e["control_frequency"])
            if any(not _close(derived[k], e["metrics"][k], AGGREGATE_RTOL, 1e-12) for k in spec.METRIC_KEYS):
                report.fail("consistency", f"{f}/{v} episode {e['episode_index']}: metrics do not follow from raw values")
                problems += 1
        reported = results.get("results", {}).get(f, {}).get(v)
        if reported is None:
            continue
        derived = res.aggregate_variation(eps)
        for k in spec.METRIC_KEYS:
            for stat in ("mean", "std"):
                a, b = derived["metrics"][k][stat], reported["metrics"][k][stat]
                if not _close(a, b, AGGREGATE_RTOL, 1e-12):
                    report.fail("consistency", f"{f}/{v} {k}.{stat}: results.json {b!r} != logs {a!r}")
                    problems += 1
    if not problems:
        n = sum(len(e) for e in episodes.values())
        report.ok("consistency", f"results.json re-derived exactly from {n} episode logs")
    return episodes


def check_provenance(results: Dict[str, Any], report: Report) -> None:
    prov = results.get("provenance", {})
    if prov.get("git_dirty"):
        report.fail("provenance", "evaluated from a RoboEval tree with modified tracked files")
    official = spec.OFFICIAL_SOURCE_FINGERPRINT
    if official is None:
        report.warn("provenance", "official source fingerprint not yet published; skipped comparison")
    elif prov.get("source_fingerprint") != official:
        report.fail("provenance", "source fingerprint differs from the official release; "
                                  "check out the release tag and re-run")
    math = prov.get("math", {})
    if math.get("pinned_env") is not None and math != math_fingerprint():
        report.warn("provenance", "evaluated in a different numeric environment "
                                  f"({math.get('machine')}, numpy {math.get('numpy')}, {math.get('blas')}, "
                                  f"pinned={math.get('pinned_env')}); replay may not match exactly")
    if results.get("harness_version") != spec.HARNESS_VERSION:
        report.warn("provenance", f"harness version {results.get('harness_version')} != {spec.HARNESS_VERSION}")
    if not any(i["check"] == "provenance" and i["level"] == "FAIL" for i in report.items):
        report.ok("provenance", f"commit {prov.get('git_describe')}, mujoco {prov.get('mujoco_version')}")


def check_replay(sub: Path, meta: Dict[str, Any], episodes: Dict[tuple, List[Dict[str, Any]]],
                 n: int, report: Report, seed: int = 0) -> None:
    """Replay logged actions through a fresh environment and compare outcomes."""
    from roboeval.challenge.submission import make_env

    # Cameras do not affect physics; replay state-only for speed.
    interface = Interface.from_dict({**meta["interface"], "observation": {"proprioception": True, "cameras": []}})
    pool = [e for eps in episodes.values() for e in eps]
    picked = random.Random(seed).sample(pool, min(n, len(pool)))
    picked.sort(key=lambda e: (e["family"], e["variation"], e["episode_index"]))
    envs: Dict[tuple, Any] = {}
    mismatches = 0
    for e in picked:
        key = (e["family"], e["variation"])
        if key not in envs:
            envs[key] = make_env(e["family"], e["variation"], interface)
        env = envs[key]
        actions = np.load(sub / "logs" / spec.variation_dirname(*key) / e["actions_file"])["actions"]
        env.reset(seed=e["seed"])
        info, success = {}, False
        for a in actions:
            _, _, _, _, info = env.step(a)
            success = env.success
        replayed = res.episode_metrics(res.to_jsonable(info), len(actions), e["control_frequency"])
        logged = e["metrics"]
        bad = [k for k in spec.METRIC_KEYS if not _close(replayed[k], logged[k], REPLAY_RTOL, REPLAY_ATOL)]
        if e["termination"] != "error" and bool(success) != (e["termination"] == "success"):
            bad.insert(0, "termination")
        if bad:
            mismatches += 1
            report.fail("replay", f"{key[0]}/{key[1]} episode {e['episode_index']}: replay differs on {', '.join(bad)}")
    for env in envs.values():
        env.close()
    if mismatches and not is_pinned():
        report.warn("replay", "this validator is not running with the pinned math environment "
                              "(run it as `python -m roboeval.challenge.validate`)")
    if not mismatches:
        report.ok("replay", f"{len(picked)} replayed episode(s) reproduce their logged metrics")


def validate(sub: Path, replay: int = 0) -> Report:
    sub = Path(sub).resolve()
    report = Report()
    check_layout(sub, report)
    meta = check_metadata(sub, report) if (sub / "metadata.yaml").exists() else None
    results_path = sub / "results.json"
    if not results_path.exists():
        return report
    try:
        results = json.loads(results_path.read_text())
    except json.JSONDecodeError as e:
        report.fail("schema", f"results.json is not valid JSON: {e}")
        return report
    check_schema(results, meta, report)
    episodes = check_consistency(sub, results, report) if (sub / "logs").is_dir() else {}
    check_provenance(results, report)
    if replay and meta and episodes:
        check_replay(sub, meta, episodes, replay, report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a challenge submission bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("submission", type=Path, help="Submission directory.")
    parser.add_argument("--replay", type=int, default=0, metavar="N",
                        help="Also replay N randomly chosen logged episodes in the simulator.")
    parser.add_argument("--json", type=Path, default=None, help="Write the report as JSON to this path.")
    args = parser.parse_args(argv)
    report = validate(args.submission, replay=args.replay)
    report.print()
    if args.json:
        args.json.write_text(json.dumps(report.items, indent=2))
    return 1 if report.failed else 0


if __name__ == "__main__":
    ensure_pinned("roboeval.challenge.validate", sys.argv[1:])
    sys.exit(main())
