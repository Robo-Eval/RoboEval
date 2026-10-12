"""Submission ingest, validation, storage, and ranking for the challenge portal.

Kept free of UI code so it can be tested and run from the command line. The
portal never executes participant code: validation reads metadata, results,
and logs, and replays logged actions in the simulator.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

from roboeval.challenge import scoring, spec
from roboeval.challenge.package import MANIFEST
from roboeval.challenge.validate import validate

ACCEPTED, REVIEW, REJECTED, HIDDEN = "accepted", "needs_review", "rejected", "hidden"
RANKED_STATUSES = (ACCEPTED,)
# Statuses that count against a team's submission limit. Bundles rejected by
# the automated checks do not, so teams can fix and resubmit.
COUNTED_STATUSES = (ACCEPTED, REVIEW, HIDDEN)


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


@dataclass
class Config:
    """Portal settings, read from environment variables (Space variables/secrets)."""

    store: str = "local:./portal_data"          # local:<dir> or hf:<namespace>/<dataset>
    hf_token: Optional[str] = None
    submissions_open: bool = True
    deadline: Optional[dt.datetime] = None      # UTC; submissions close after it
    limit_per_day: Optional[int] = 2            # per team, UTC day; None = unlimited
    limit_total: Optional[int] = None           # per team over the challenge
    max_archive_gb: float = 5.0
    store_archive_max_gb: float = 2.0           # keep a copy for audits below this size
    replay_episodes: int = 20
    replay_mismatch: str = "review"             # "review" or "reject"
    organizers: Tuple[str, ...] = ()            # HF usernames: admin tab, marked not prize-eligible

    @classmethod
    def from_env(cls) -> "Config":
        def opt_int(name: str, default: Optional[int]) -> Optional[int]:
            v = os.environ.get(name)
            if v is None:
                return default
            return None if v.strip().lower() in ("", "none", "unlimited") else int(v)

        deadline = os.environ.get("SUBMISSION_DEADLINE")
        return cls(
            store=os.environ.get("PORTAL_STORE", cls.store),
            hf_token=os.environ.get("HF_TOKEN"),
            submissions_open=_env_bool("SUBMISSIONS_OPEN", True),
            deadline=dt.datetime.fromisoformat(deadline).astimezone(dt.timezone.utc) if deadline else None,
            limit_per_day=opt_int("SUBMISSION_LIMIT_PER_DAY", cls.limit_per_day),
            limit_total=opt_int("SUBMISSION_LIMIT_TOTAL", cls.limit_total),
            max_archive_gb=float(os.environ.get("MAX_ARCHIVE_GB", cls.max_archive_gb)),
            store_archive_max_gb=float(os.environ.get("STORE_ARCHIVE_MAX_GB", cls.store_archive_max_gb)),
            replay_episodes=int(os.environ.get("REPLAY_EPISODES", cls.replay_episodes)),
            replay_mismatch=os.environ.get("REPLAY_MISMATCH", cls.replay_mismatch),
            organizers=tuple(u.strip() for u in os.environ.get("ORGANIZERS", "").split(",") if u.strip()),
        )


class SubmissionError(Exception):
    """A problem the participant must fix; the message is shown to them."""


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ─────────────────────────────── storage ────────────────────────────────
class Store:
    """Submission records: submissions/<id>/{record.json, results.json, ...}."""

    def put_submission(self, sid: str, files: Dict[str, Path]) -> None:
        raise NotImplementedError

    def put_record(self, sid: str, record: Dict[str, Any]) -> None:
        raise NotImplementedError

    def records(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def read_json(self, sid: str, name: str) -> Dict[str, Any]:
        raise NotImplementedError


class LocalStore(Store):
    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "submissions").mkdir(parents=True, exist_ok=True)

    def put_submission(self, sid: str, files: Dict[str, Path]) -> None:
        d = self.root / "submissions" / sid
        d.mkdir(parents=True, exist_ok=True)
        for name, src in files.items():
            shutil.copyfile(src, d / name)

    def put_record(self, sid: str, record: Dict[str, Any]) -> None:
        d = self.root / "submissions" / sid
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "record.json.tmp"
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(d / "record.json")

    def records(self) -> List[Dict[str, Any]]:
        return [json.loads(p.read_text()) for p in sorted((self.root / "submissions").glob("*/record.json"))]

    def read_json(self, sid: str, name: str) -> Dict[str, Any]:
        return json.loads((self.root / "submissions" / sid / name).read_text())


class HFStore(Store):
    """A (private) Hugging Face dataset repo; one commit per submission."""

    def __init__(self, repo_id: str, token: Optional[str]):
        from huggingface_hub import HfApi

        self.repo_id, self.token = repo_id, token
        self.api = HfApi(token=token)
        self.api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    def put_submission(self, sid: str, files: Dict[str, Path]) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name, src in files.items():
                shutil.copyfile(src, Path(tmp) / name)
            self.api.upload_folder(repo_id=self.repo_id, repo_type="dataset", folder_path=tmp,
                                   path_in_repo=f"submissions/{sid}", commit_message=f"submission {sid}")

    def put_record(self, sid: str, record: Dict[str, Any]) -> None:
        self.api.upload_file(repo_id=self.repo_id, repo_type="dataset",
                             path_or_fileobj=io.BytesIO(json.dumps(record, indent=2).encode()),
                             path_in_repo=f"submissions/{sid}/record.json", commit_message=f"record {sid}")

    def records(self) -> List[Dict[str, Any]]:
        from huggingface_hub import snapshot_download

        local = snapshot_download(self.repo_id, repo_type="dataset", token=self.token,
                                  allow_patterns=["submissions/*/record.json"])
        return [json.loads(p.read_text()) for p in sorted(Path(local).glob("submissions/*/record.json"))]

    def read_json(self, sid: str, name: str) -> Dict[str, Any]:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(self.repo_id, f"submissions/{sid}/{name}", repo_type="dataset", token=self.token)
        return json.loads(Path(path).read_text())


def make_store(cfg: Config) -> Store:
    kind, _, target = cfg.store.partition(":")
    if kind == "local":
        return LocalStore(Path(target))
    if kind == "hf":
        return HFStore(target, cfg.hf_token)
    raise ValueError(f"PORTAL_STORE must be local:<dir> or hf:<repo>, got {cfg.store!r}")


# ─────────────────────────────── archives ───────────────────────────────
def _normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith("https://"):
        raise SubmissionError("The archive link must be an https:// URL.")
    # A Hugging Face file page (…/blob/…) -> its download URL (…/resolve/…).
    return re.sub(r"^(https://huggingface\.co/.+?)/blob/", r"\1/resolve/", url)


def fetch_archive(source: str, dest: Path, max_bytes: int) -> None:
    """Copy an uploaded file or stream a URL to ``dest``, enforcing a size cap."""
    if Path(source).is_file():
        if Path(source).stat().st_size > max_bytes:
            raise SubmissionError(f"Archive exceeds the {max_bytes / 1e9:.1f} GB limit.")
        shutil.copyfile(source, dest)
        return
    try:
        with requests.get(_normalize_url(source), stream=True, timeout=60, allow_redirects=True) as r:
            if r.status_code != 200:
                raise SubmissionError(f"Downloading the archive returned HTTP {r.status_code}. "
                                      "The link must be downloadable without signing in.")
            size = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    size += len(chunk)
                    if size > max_bytes:
                        raise SubmissionError(f"Archive exceeds the {max_bytes / 1e9:.1f} GB limit.")
                    f.write(chunk)
    except requests.RequestException as e:
        raise SubmissionError(f"Could not download the archive: {e}") from e


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def safe_extract(archive: Path, dest: Path, max_bytes: int, max_members: int = 20000) -> Dict[str, str]:
    """Extract submission/ from a .tar.gz without trusting its contents.

    Rejects links, devices, absolute or escaping paths, and anything outside
    submission/. Checkpoint files are hashed but written as empty placeholders:
    validation does not need the weights, and they can be many GB.
    Returns {relative path: sha256} for every regular file.
    """
    hashes: Dict[str, str] = {}
    total = 0
    root = dest / "submission"
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for n, member in enumerate(tar):
                if n >= max_members:
                    raise SubmissionError(f"Archive has more than {max_members} entries.")
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "submission":
                    raise SubmissionError(f"Unexpected path in archive: {member.name!r} "
                                          "(everything must be under submission/).")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise SubmissionError(f"Archive entry {member.name!r} is a link or special file.")
                rel = PurePosixPath(*path.parts[1:]).as_posix()
                total += member.size
                if total > max_bytes:
                    raise SubmissionError("Archive expands beyond the size limit.")
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                src = tar.extractfile(member)
                skip = rel.startswith("checkpoint/")
                with open(target, "wb") as out:
                    while block := src.read(1 << 20):
                        h.update(block)
                        if not skip:
                            out.write(block)
                hashes[rel] = h.hexdigest()
    except (tarfile.TarError, EOFError, OSError) as e:
        raise SubmissionError(f"Archive is not a readable .tar.gz: {e}") from e
    if not hashes:
        raise SubmissionError("Archive is empty.")
    return hashes


def read_manifest(sub: Path) -> Dict[str, str]:
    listed = {}
    for line in (sub / MANIFEST).read_text().splitlines():
        if line.strip():
            digest, _, rel = line.partition("  ")
            listed[rel] = digest
    return listed


def verify_manifest(sub: Path, hashes: Dict[str, str]) -> None:
    if MANIFEST not in hashes:
        raise SubmissionError("MANIFEST.sha256 is missing; build the archive with "
                              "`python -m roboeval.challenge.package`.")
    listed = read_manifest(sub)
    actual = {k: v for k, v in hashes.items() if k != MANIFEST}
    altered = sorted(k for k in listed if k in actual and actual[k] != listed[k])
    missing = sorted(k for k in listed if k not in actual)
    unlisted = sorted(k for k in actual if k not in listed)
    problems = []
    if altered:
        problems.append(f"{len(altered)} file(s) differ from the manifest, e.g. {altered[0]}")
    if missing:
        problems.append(f"{len(missing)} listed file(s) missing, e.g. {missing[0]}")
    if unlisted:
        problems.append(f"{len(unlisted)} file(s) not in the manifest, e.g. {unlisted[0]}")
    if problems:
        raise SubmissionError("Archive does not match MANIFEST.sha256: " + "; ".join(problems)
                              + ". Re-run `python -m roboeval.challenge.package`.")


# ─────────────────────────────── policy ────────────────────────────────
def team_of(records: List[Dict[str, Any]], user: str) -> Optional[str]:
    for r in sorted(records, key=lambda r: r["created_utc"]):
        if r["hf_user"] == user and r["status"] != REJECTED:
            return r["team_name"]
    return None


def owner_of(records: List[Dict[str, Any]], team: str) -> Optional[str]:
    for r in sorted(records, key=lambda r: r["created_utc"]):
        if r["team_name"].casefold() == team.casefold() and r["status"] != REJECTED:
            return r["hf_user"]
    return None


def check_duplicate(records: List[Dict[str, Any]], user: str, sha256: str) -> None:
    """Refuse an archive this account already has accepted or under review.

    Guards against double clicks, reloads, and second tabs. Rejected archives may
    be resubmitted, e.g. after a validator fix.
    """
    for r in records:
        if r["hf_user"] == user and r["archive_sha256"] == sha256 and r["status"] in (ACCEPTED, REVIEW):
            raise SubmissionError(f"This archive was already submitted as `{r['id']}` (status: {r['status']}); "
                                  "see the My submissions tab. It was not submitted again.")


def check_window(cfg: Config) -> None:
    if not cfg.submissions_open:
        raise SubmissionError("Submissions are closed.")
    if cfg.deadline and now_utc() > cfg.deadline:
        raise SubmissionError(f"The submission deadline ({cfg.deadline.isoformat()}) has passed.")


def check_limits(cfg: Config, records: List[Dict[str, Any]], team: str) -> None:
    counted = [r for r in records if r["team_name"].casefold() == team.casefold() and r["status"] in COUNTED_STATUSES]
    today = now_utc().date().isoformat()
    if cfg.limit_per_day is not None and sum(r["created_utc"][:10] == today for r in counted) >= cfg.limit_per_day:
        raise SubmissionError(f"Team '{team}' has used its {cfg.limit_per_day} submission(s) for today (UTC).")
    if cfg.limit_total is not None and len(counted) >= cfg.limit_total:
        raise SubmissionError(f"Team '{team}' has used all {cfg.limit_total} submissions.")


def check_identity(records: List[Dict[str, Any]], user: str, team: str) -> None:
    """One HF account per team, and one team per HF account."""
    mine = team_of(records, user)
    if mine is not None and mine.casefold() != team.casefold():
        raise SubmissionError(f"Your account already submits for team '{mine}'; "
                              f"metadata.yaml says '{team}'. Each person may belong to one team.")
    owner = owner_of(records, team)
    if owner is not None and owner != user:
        raise SubmissionError(f"Team '{team}' is registered to another account. "
                              "Each team submits under a single identity.")


# ─────────────────────────────── ingest ────────────────────────────────
@dataclass
class Outcome:
    record: Dict[str, Any]
    report: List[Dict[str, str]] = field(default_factory=list)


def process_submission(cfg: Config, store: Store, user: str, source: str, expected_sha256: str) -> Outcome:
    """Fetch, verify, validate, and record one submission.

    Raises SubmissionError for problems found before validation (nothing is
    recorded); validation failures are recorded as rejected so the team can
    see the report in their history.
    """
    check_window(cfg)
    expected = expected_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise SubmissionError("Paste the 64-character SHA-256 printed by `package`.")
    records = store.records()
    check_duplicate(records, user, expected)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        archive = tmp / "bundle.tar.gz"
        fetch_archive(source, archive, int(cfg.max_archive_gb * 1e9))
        actual = sha256_file(archive)
        if actual != expected:
            raise SubmissionError(f"SHA-256 mismatch: the downloaded archive is {actual}. "
                                  "Check the link points at the archive you packaged.")
        hashes = safe_extract(archive, tmp / "x", int(cfg.max_archive_gb * 4e9))
        sub = tmp / "x" / "submission"
        verify_manifest(sub, hashes)
        try:
            meta = yaml.safe_load((sub / "metadata.yaml").read_text()) or {}
        except (OSError, yaml.YAMLError) as e:
            raise SubmissionError(f"metadata.yaml is unreadable: {e}") from e
        team = str(meta.get("team_name", "")).strip()
        method = str(meta.get("method_name", "")).strip()
        if not team or not method:
            raise SubmissionError("metadata.yaml must set team_name and method_name.")

        check_identity(records, user, team)
        check_limits(cfg, records, team)

        report = validate(sub, replay=cfg.replay_episodes)
        fails = [i for i in report.items if i["level"] == "FAIL"]
        if not fails:
            status = ACCEPTED
        elif all(i["check"] == "replay" for i in fails) and cfg.replay_mismatch == "review":
            status = REVIEW
        else:
            status = REJECTED

        results = json.loads((sub / "results.json").read_text()) if (sub / "results.json").exists() else {}
        sid = f"{now_utc().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        record = {
            "id": sid,
            "created_utc": now_utc().isoformat(timespec="seconds"),
            "hf_user": user,
            "team_name": team,
            "method_name": method,
            "affiliation": meta.get("affiliation"),
            "contact_email": meta.get("contact_email"),
            "organizer": user in cfg.organizers,
            "open_source": bool(meta.get("open_source")),
            "code_url": meta.get("code_url") or None,
            "checkpoint_url": meta.get("checkpoint_url") or None,
            "observation_mode": "vision" if (meta.get("interface", {}).get("observation", {}) or {}).get("cameras") else "state",
            "interface": results.get("interface"),
            "source": source if not Path(source).is_file() else "upload",
            "archive_sha256": actual,
            "status": status,
            "validation": report.items,
            "summary": results.get("summary", {}) if status != REJECTED else {},
            "provenance": results.get("provenance", {}),
        }
        files = {"metadata.yaml": sub / "metadata.yaml", "validation.json": tmp / "validation.json"}
        (tmp / "validation.json").write_text(json.dumps(report.items, indent=2))
        if (sub / "results.json").exists():
            files["results.json"] = sub / "results.json"
        if archive.stat().st_size <= cfg.store_archive_max_gb * 1e9:
            files["bundle.tar.gz"] = archive
        store.put_submission(sid, files)
        store.put_record(sid, record)
        return Outcome(record=record, report=report.items)


def set_status(store: Store, records: List[Dict[str, Any]], sid: str, status: str, note: str = "") -> Dict[str, Any]:
    """Organizer decision on a submission (approve a review, hide a disqualified entry, ...)."""
    rec = next((r for r in records if r["id"] == sid), None)
    if rec is None:
        raise SubmissionError(f"No submission {sid}.")
    rec = {**rec, "status": status, "decision_note": note, "decided_utc": now_utc().isoformat(timespec="seconds")}
    store.put_record(sid, rec)
    return rec


# ─────────────────────────────── ranking ───────────────────────────────
METRIC_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "Outcome": ("success", "tp"),
    "Efficiency": ("cpl", "jpl", "opl", "ct", "tl"),
    "Coordination": ("bavd", "bgvd"),
    "Safety & Stability": ("scc", "ecc", "sc", "mjj", "mcj"),
}
HIGHER_IS_BETTER = ("success", "tp")
ACTION_MODES = ("EE delta", "EE absolute", "Joint delta", "Joint absolute")


def action_label(record: Dict[str, Any]) -> str:
    mode = ((record.get("interface") or {}).get("action_mode") or {})
    return f"{'EE' if mode.get('ee') else 'Joint'} {'absolute' if mode.get('absolute') else 'delta'}"


FAMILY_SCORE_KEYS = {"Outcome": "outcome", "Efficiency": "efficiency", "Coordination": "coordination",
                     "Safety & Stability": "safety"}


def record_scores(record: Dict[str, Any], task: Optional[str]) -> Dict[str, float]:
    """RDS and family scores for a submission, over the suite or one task."""
    scores = scoring.suite_scores(record["summary"]["per_family"])
    return scores["overall"] if task is None else scores["tasks"][task]


def task_metrics(record: Dict[str, Any], task: Optional[str]) -> Dict[str, float]:
    """Suite means (task=None) or one task family's means for a submission."""
    summary = record["summary"]
    return summary["suite"] if task is None else summary["per_family"][task]


def leaderboard(records: List[Dict[str, Any]], task: Optional[str] = None) -> List[Dict[str, Any]]:
    """Most recent accepted submission per team, over the suite or one task.

    Ranked by the RoboEval Diagnostic Score, then Success, Task Progression,
    and earlier submission, as the rules' tie-break order specifies.
    """
    latest: Dict[str, Dict[str, Any]] = {}
    for r in sorted(records, key=lambda r: r["created_utc"]):
        if r["status"] in RANKED_STATUSES:
            latest[r["team_name"].casefold()] = r
    rows = [r for r in latest.values() if task is None or task in r["summary"].get("per_family", {})]
    rows.sort(key=lambda r: (-record_scores(r, task)["score"], -task_metrics(r, task)["success"],
                             -task_metrics(r, task)["tp"], -record_scores(r, task)["safety"], r["created_utc"]))
    return rows


def leaderboard_table(records: List[Dict[str, Any]], observation_mode: Optional[str] = None,
                      task: Optional[str] = None, family: Optional[str] = None,
                      action_mode: Optional[str] = None) -> List[Dict[str, Any]]:
    """Rows for one leaderboard view; ranks are within the filtered view."""
    keys = METRIC_FAMILIES[family] if family else spec.METRIC_KEYS
    out = []
    for r in leaderboard(records, task):
        if observation_mode and r["observation_mode"] != observation_mode:
            continue
        if action_mode and action_label(r) != action_mode:
            continue
        m, sc = task_metrics(r, task), record_scores(r, task)
        row = {"Rank": len(out) + 1, "Team": r["team_name"], "Method": r["method_name"],
               "Obs.": r["observation_mode"], "Action": action_label(r), "SCORE ↑": round(sc["score"], 3)}
        if family:
            row[f"{family.upper()} SCORE ↑"] = round(sc[FAMILY_SCORE_KEYS[family]], 3)
        row.update({f"{k.upper()} {'↑' if k in HIGHER_IS_BETTER else '↓'}": round(m[k], 3) for k in keys})
        row["Open source"] = "yes" if r["open_source"] else ""
        row["Note"] = "organizer (not prize-eligible)" if r["organizer"] else ""
        row["Submitted (UTC)"] = r["created_utc"][:16].replace("T", " ")
        out.append(row)
    return out


# Website keys for static/data/leaderboard_by_task.json.
WEBSITE_FAMILY_NAMES = {
    "CubeHandover": "Cube Handover",
    "LiftPot": "Lift Pot",
    "LiftTray": "Lift Tray",
    "PackBox": "Pack Box",
    "PickSingleBookFromTable": "Pick Single Book From Table",
    "RotateValve": "Rotate Valve",
    "StackSingleBookShelf": "Stack Single Book Shelf",
    "StackTwoBlocks": "Stack Two Blocks",
}


def website_export(records: List[Dict[str, Any]], results_by_id: Dict[str, Dict[str, Any]]) -> Tuple[list, dict]:
    """Ranked entries in the website's leaderboard JSON shapes (global, by task).

    Values are means over variations; ``std`` is the mean of per-variation stds,
    matching how the existing site files report baselines.
    """
    import numpy as np

    global_rows, by_task = [], {name: [] for name in WEBSITE_FAMILY_NAMES.values()}
    for r in leaderboard(records):
        res = results_by_id[r["id"]]["results"]
        label = f"{r['method_name']} ({r['team_name']})"

        def agg(entries):
            return {k: {"mean": float(np.mean([e["metrics"][k]["mean"] for e in entries])),
                        "std": float(np.mean([e["metrics"][k]["std"] for e in entries]))} for k in spec.METRIC_KEYS}

        global_rows.append({"method": label, "metrics": agg([v for vs in res.values() for v in vs.values()])})
        for fam, vs in res.items():
            by_task[WEBSITE_FAMILY_NAMES[fam]].append({"method": label, "metrics": agg(list(vs.values()))})
    return global_rows, by_task
