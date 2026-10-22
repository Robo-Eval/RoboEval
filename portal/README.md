---
title: RoboEval Challenge CoRL 2026
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
hf_oauth: true
pinned: true
short_description: Submissions and leaderboard for the RoboEval bimanual track
---

# RoboEval challenge portal

Submission portal and leaderboard for the CoRL 2026 RoboEval Diagnostic
Bimanual Challenge. Participants sign in with Hugging Face, submit the archive
built by `python -m roboeval.challenge.package`, and get the validation report
back in a few minutes.

## What happens to a submission

1. **Gate:** submissions open, before the deadline, a 64-hex SHA-256 given.
2. **Fetch:** the archive is downloaded from its https link (or taken from the
   upload) up to `MAX_ARCHIVE_GB`, and must match the given SHA-256.
3. **Unpack safely:** only regular files under `submission/`; no links,
   absolute or `..` paths, or archives that expand past the limit. Checkpoint
   files are hashed but not stored.
4. **Manifest:** every file must match `MANIFEST.sha256`, with nothing extra.
5. **Identity:** one Hugging Face account per team and one team per account
   (the first accepted submission binds them). Then the team's limits apply.
6. **Validation:** `roboeval.challenge.validate` with `REPLAY_EPISODES` replays,
   on the same release participants use, in the same pinned numeric
   environment. No participant code is executed.
7. **Decision:** all checks pass → **accepted** (on the leaderboard). Only the
   replay fails → **needs_review** (or rejected, per `REPLAY_MISMATCH`).
   Anything else → **rejected**; rejected bundles do not count toward limits.

Everything is stored in a private dataset repo: per submission, `record.json`,
`results.json`, `metadata.yaml`, `validation.json`, `report.pdf`, and the archive
itself when it is under `STORE_ARCHIVE_MAX_GB` (for audits and held-out re-runs).
A larger archive is kept without its `checkpoint/` weights as
`bundle-no-weights.tar.gz` (report, policy code, logs) when that is under
`STORE_LITE_MAX_GB`; if the team gave no `checkpoint_url`, the report warns them
to upload the weights to Hugging Face and send the link.

The leaderboard shows each team's most recent accepted submission, ranked by
the RoboEval Diagnostic Score (`roboeval/challenge/scoring.py`), then Success,
Task Progression, the safety score, and submission time. It has per-task and
per-metric-family views and filters by observation and action mode.
Organizers' own entries are labelled not prize-eligible.

## Configuration

Space **secrets**: `HF_TOKEN`, a token with write access to the store dataset.

Space **variables**:

| variable | default | meaning |
|---|---|---|
| `PORTAL_STORE` | `local:./portal_data` | `hf:<namespace>/<dataset>` in production (created private if missing) |
| `SUBMISSIONS_OPEN` | `true` | master switch |
| `SUBMISSION_DEADLINE` | unset | ISO time, e.g. `2026-11-01T23:59:00-07:00` |
| `SUBMISSION_LIMIT_PER_DAY` | `2` | per team per UTC day; `none` for unlimited |
| `SUBMISSION_LIMIT_TOTAL` | unset | per team over the challenge |
| `MAX_ARCHIVE_GB` | `5` | download and upload cap |
| `STORE_ARCHIVE_MAX_GB` | `2` | keep a copy of archives up to this size |
| `STORE_LITE_MAX_GB` | `1` | above that, keep the archive minus `checkpoint/` up to this size |
| `REPLAY_EPISODES` | `20` | episodes replayed per submission |
| `REPLAY_MISMATCH` | `review` | `review` or `reject` |
| `ORGANIZERS` | unset | comma-separated HF usernames: Organizers tab, entries marked not prize-eligible |

The Docker build arguments `ROBOEVAL_REPO` and `ROBOEVAL_REF` choose the
RoboEval release the validator runs; keep `ROBOEVAL_REF` on the challenge tag.

## Deploying

The Space repo holds the contents of this `portal/` directory.

```bash
hf repo create <namespace>/roboeval-challenge --repo-type space --space-sdk docker
hf upload <namespace>/roboeval-challenge portal/ . --repo-type space
```

Then set the secret and variables above in the Space settings. CPU basic
hardware is enough: a submission validates in a few minutes.

## Organizer tasks

- **Review queue:** the Organizers tab lists every submission. Approve a
  `needs_review` entry by setting it to `accepted`, or hide a disqualified
  one with `hidden`; the note is kept in its record.
- **Held-out re-runs:** download the archive from the store and run
  `python -m roboeval.challenge.evaluate --submission <dir> --seeds-file heldout.json --output-dir <out>`
  (the checkpoint comes from the participant's `checkpoint_url` or a direct request).
- **Website:** `python export_leaderboard.py --out <site>/static/data` writes
  `challenge_leaderboard_*.json` in the site's leaderboard format.

## Local development

```bash
pip install -e <RoboEval checkout> -r requirements.txt
OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 python app.py   # store defaults to ./portal_data
PORTAL_TEST_BUNDLE=<archive.tar.gz> pytest test_portal.py
```
