"""Portal tests. Needs a valid archive from `python -m roboeval.challenge.package`:

    PORTAL_TEST_BUNDLE=/path/to/Team_Method.tar.gz pytest portal/test_portal.py
"""
import datetime as dt
import functools
import hashlib
import http.server
import io
import json
import os
import tarfile
import threading
from pathlib import Path

import pytest

import portal_core as core

BUNDLE = os.environ.get("PORTAL_TEST_BUNDLE")
pytestmark = pytest.mark.skipif(not BUNDLE or not Path(BUNDLE).is_file(), reason="set PORTAL_TEST_BUNDLE")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def cfg(tmp_path):
    return core.Config(store=f"local:{tmp_path / 'store'}", replay_episodes=3, limit_per_day=5)


@pytest.fixture
def store(cfg):
    return core.make_store(cfg)


def repack(src: Path, dst: Path, edit=None, extra=None, rewrite_manifest=False, drop=()) -> Path:
    """Copy an archive, optionally editing (name -> bytes -> bytes) or dropping members by prefix."""
    files = {}
    with tarfile.open(src, "r:gz") as t:
        for m in t:
            if m.isfile() and not any(m.name.startswith(d) for d in drop):
                files[m.name] = t.extractfile(m).read()
    for name, fn in (edit or {}).items():
        files[name] = fn(files[name])
    if rewrite_manifest:
        lines = [f"{hashlib.sha256(b).hexdigest()}  {n[len('submission/'):]}"
                 for n, b in sorted(files.items()) if not n.endswith("MANIFEST.sha256")]
        files["submission/MANIFEST.sha256"] = ("\n".join(lines) + "\n").encode()
    with tarfile.open(dst, "w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
        for info, data in (extra or []):
            t.addfile(info, io.BytesIO(data) if data is not None else None)
    return dst


def test_valid_bundle_is_accepted_and_ranked(cfg, store):
    out = core.process_submission(cfg, store, "alice", BUNDLE, sha(BUNDLE))
    assert out.record["status"] == core.ACCEPTED, out.report
    rows = core.leaderboard_table(store.records())
    assert rows[0]["Team"] == out.record["team_name"] and rows[0]["Rank"] == 1
    assert (Path(cfg.store[6:]) / "submissions" / out.record["id"] / "bundle.tar.gz").is_file()


def test_checkpoint_hosted_elsewhere_is_accepted(cfg, store, tmp_path):
    """package --exclude-checkpoint leaves checkpoint/ out when checkpoint_url is set."""
    hosted = repack(Path(BUNDLE), tmp_path / "hosted.tar.gz", rewrite_manifest=True, drop=("submission/checkpoint/",),
                    edit={"submission/metadata.yaml": lambda b: b.replace(
                        b'checkpoint_url: ""', b'checkpoint_url: "https://huggingface.co/example/weights"')})
    out = core.process_submission(cfg, store, "alice", str(hosted), sha(hosted))
    assert out.record["status"] == core.ACCEPTED, out.report


def test_sha_mismatch_rejected_before_recording(cfg, store):
    with pytest.raises(core.SubmissionError, match="SHA-256 mismatch"):
        core.process_submission(cfg, store, "alice", BUNDLE, "0" * 64)
    assert store.records() == []


def test_edited_file_breaks_manifest(cfg, store, tmp_path):
    bad = repack(Path(BUNDLE), tmp_path / "bad.tar.gz",
                 edit={"submission/results.json": lambda b: b.replace(b'"success"', b'"success" ', 1)})
    with pytest.raises(core.SubmissionError, match="MANIFEST"):
        core.process_submission(cfg, store, "alice", str(bad), sha(bad))


def test_edited_results_with_regenerated_manifest_is_rejected(cfg, store, tmp_path):
    def inflate(b):
        r = json.loads(b)
        r["results"]["LiftPot"]["PositionAndOrientation"]["metrics"]["success"]["mean"] = 0.99
        return json.dumps(r).encode()

    bad = repack(Path(BUNDLE), tmp_path / "bad.tar.gz", edit={"submission/results.json": inflate},
                 rewrite_manifest=True)
    out = core.process_submission(cfg, store, "alice", str(bad), sha(bad))
    assert out.record["status"] == core.REJECTED
    assert any(i["check"] == "consistency" and i["level"] == "FAIL" for i in out.report)
    assert core.leaderboard(store.records()) == []


@pytest.mark.parametrize("name,kind", [("../evil.txt", tarfile.REGTYPE), ("/etc/evil", tarfile.REGTYPE),
                                       ("submission/link", tarfile.SYMTYPE), ("other/file", tarfile.REGTYPE)])
def test_unsafe_members_rejected(cfg, store, tmp_path, name, kind):
    info = tarfile.TarInfo(name)
    info.type = kind
    data = b"x"
    if kind == tarfile.SYMTYPE:
        info.linkname, data = "/etc/passwd", None
    else:
        info.size = 1
    bad = repack(Path(BUNDLE), tmp_path / "bad.tar.gz", extra=[(info, data)])
    with pytest.raises(core.SubmissionError):
        core.process_submission(cfg, store, "alice", str(bad), sha(bad))
    assert not (tmp_path.parent / "evil.txt").exists()


def test_team_identity_rules(cfg, store, tmp_path):
    team = core.process_submission(cfg, store, "alice", BUNDLE, sha(BUNDLE)).record["team_name"]
    with pytest.raises(core.SubmissionError, match="registered to another account"):
        core.process_submission(cfg, store, "mallory", BUNDLE, sha(BUNDLE))
    other = repack(Path(BUNDLE), tmp_path / "other.tar.gz", rewrite_manifest=True,
                   edit={"submission/metadata.yaml": lambda b: b.replace(team.encode(), b"Second Team")})
    with pytest.raises(core.SubmissionError, match="already submits for team"):
        core.process_submission(cfg, store, "alice", str(other), sha(other))


def test_daily_limit_counts_only_accepted(cfg, store, tmp_path):
    cfg.limit_per_day = 1
    bad = repack(Path(BUNDLE), tmp_path / "bad.tar.gz", rewrite_manifest=True,
                 edit={"submission/results.json": lambda b: b.replace(b'"seeds"', b'"seeds_x"', 1)})
    assert core.process_submission(cfg, store, "alice", str(bad), sha(bad)).record["status"] == core.REJECTED
    assert core.process_submission(cfg, store, "alice", BUNDLE, sha(BUNDLE)).record["status"] == core.ACCEPTED
    with pytest.raises(core.SubmissionError, match="for today"):
        core.process_submission(cfg, store, "alice", BUNDLE, sha(BUNDLE))


def test_closed_and_past_deadline(cfg, store):
    cfg.deadline = core.now_utc() - dt.timedelta(minutes=1)
    with pytest.raises(core.SubmissionError, match="deadline"):
        core.process_submission(cfg, store, "alice", BUNDLE, sha(BUNDLE))
    cfg.deadline, cfg.submissions_open = None, False
    with pytest.raises(core.SubmissionError, match="closed"):
        core.process_submission(cfg, store, "alice", BUNDLE, sha(BUNDLE))


def test_download_from_url_and_http_errors(cfg, store, monkeypatch):
    directory = str(Path(BUNDLE).parent)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        # fetch_archive requires https; exercise the downloader directly over http.
        dest = Path(cfg.store[6:]) / "dl.tar.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(core, "_normalize_url", lambda u: u)
        core.fetch_archive(f"http://127.0.0.1:{server.server_port}/{Path(BUNDLE).name}", dest, 10**9)
        assert sha(dest) == sha(BUNDLE)
        with pytest.raises(core.SubmissionError, match="HTTP 404"):
            core.fetch_archive(f"http://127.0.0.1:{server.server_port}/missing.tar.gz", dest, 10**9)
        with pytest.raises(core.SubmissionError, match="limit"):
            core.fetch_archive(f"http://127.0.0.1:{server.server_port}/{Path(BUNDLE).name}", dest, 1000)
    finally:
        server.shutdown()


def test_admin_status_and_latest_accepted_ranked(cfg, store):
    first = core.process_submission(cfg, store, "alice", BUNDLE, sha(BUNDLE)).record
    second = core.process_submission(cfg, store, "alice", BUNDLE, sha(BUNDLE)).record
    assert core.leaderboard(store.records())[0]["id"] == second["id"]
    core.set_status(store, store.records(), second["id"], core.HIDDEN, "test")
    assert core.leaderboard(store.records())[0]["id"] == first["id"]


def test_https_required_and_hf_blob_links():
    with pytest.raises(core.SubmissionError, match="https"):
        core._normalize_url("http://example.com/x.tar.gz")
    assert core._normalize_url("https://huggingface.co/datasets/a/b/blob/main/x.tar.gz") == \
        "https://huggingface.co/datasets/a/b/resolve/main/x.tar.gz"
