"""CoRL 2026 RoboEval challenge portal (Hugging Face Space).

Participants sign in with Hugging Face, submit the archive produced by
`python -m roboeval.challenge.package`, and get the validation report back.
Accepted submissions appear on the leaderboard; replay mismatches wait for an
organizer. All settings come from environment variables (see README.md).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

import gradio as gr
import pandas as pd

import portal_core as core
from roboeval.challenge import spec

if os.environ.get("SPACE_ID") and os.environ.get("SYSTEM") != "spaces":
    # Gradio would mock OAuth and sign every visitor in as the HF_TOKEN owner.
    raise SystemExit("Refusing to start: set SYSTEM=spaces so Gradio uses real Hugging Face sign-in.")

CFG = core.Config.from_env()
STORE = core.make_store(CFG)

_cache: Dict[str, Any] = {"t": 0.0, "records": []}
_cache_lock = threading.Lock()


def records(fresh: bool = False) -> List[Dict[str, Any]]:
    with _cache_lock:
        if fresh or time.time() - _cache["t"] > 30:
            _cache["records"], _cache["t"] = STORE.records(), time.time()
        return list(_cache["records"])


def is_organizer(profile: Optional[gr.OAuthProfile]) -> bool:
    return profile is not None and profile.username in CFG.organizers


def report_markdown(items: List[Dict[str, str]]) -> str:
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    rows = "\n".join(f"| {icon[i['level']]} | {i['check']} | {i['message']} |" for i in items)
    return "| | check | result |\n|---|---|---|\n" + rows


STATUS_TEXT = {
    core.ACCEPTED: "✅ **Accepted.** Your entry is on the leaderboard.",
    core.REVIEW: "🕵️ **Pending organizer review.** Every automated check passed except the replay, which can differ "
                 "slightly across machines. Organizers will re-run it; no action needed unless they contact you.",
    core.REJECTED: "❌ **Rejected by the automated checks.** Fix the FAIL items below and resubmit; "
                   "rejected bundles do not count toward your submission limit.",
}


# ─────────────────────────────── handlers ───────────────────────────────
TASK_CHOICES = [("Overall (all 8 tasks)", "overall")] + [(f.display_name, f.key) for f in spec.TASK_FAMILIES]
FAMILY_CHOICES = ["All metrics", *core.METRIC_FAMILIES]
ACTION_CHOICES = ["All", *core.ACTION_MODES]


def show_leaderboard(task: str, family: str, obs: str, action: str):
    rows = core.leaderboard_table(
        records(),
        observation_mode=None if obs == "all" else obs,
        task=None if task == "overall" else task,
        family=None if family == "All metrics" else family,
        action_mode=None if action == "All" else action,
    )
    return pd.DataFrame(rows or [{"Team": "No accepted submissions in this view yet"}])


def submit(url: str, upload, sha256: str, confirmed: bool, profile: Optional[gr.OAuthProfile]):
    """Generator: shows progress at once, then the validation report."""
    if profile is None:
        yield "Sign in with Hugging Face first."
        return
    if not confirmed:
        yield "Confirm that you have read and followed the rules."
        return
    source = upload if upload else (url or "").strip()
    if not source:
        yield "Provide a download link to your archive, or upload it."
        return
    yield ("⏳ **Validating your submission.** Downloading the archive, re-deriving the results from the logs, "
           f"and replaying {CFG.replay_episodes} episodes in the simulator; this usually takes 2-5 minutes. "
           "Keep this page open. If another submission is being validated, yours waits in line first.")
    try:
        outcome = core.process_submission(CFG, STORE, profile.username, source, sha256)
    except core.SubmissionError as e:
        yield f"❌ **Not submitted:** {e}"
        return
    finally:
        records(fresh=True)
    r = outcome.record
    yield (f"{STATUS_TEXT[r['status']]}\n\nSubmission `{r['id']}` · team **{r['team_name']}** · "
           f"method **{r['method_name']}**\n\n{report_markdown(outcome.report)}")


def my_submissions(profile: Optional[gr.OAuthProfile]):
    if profile is None:
        return pd.DataFrame(), "Sign in to see your submissions."
    mine = [r for r in records() if r["hf_user"] == profile.username]
    mine.sort(key=lambda r: r["created_utc"], reverse=True)
    table = [{"ID": r["id"], "Submitted (UTC)": r["created_utc"][:16].replace("T", " "), "Team": r["team_name"],
              "Method": r["method_name"], "Status": r["status"],
              "Success": round(r["summary"]["suite"]["success"], 3) if r.get("summary", {}).get("suite") else ""}
             for r in mine]
    detail = report_markdown(mine[0]["validation"]) if mine else "No submissions yet."
    return pd.DataFrame(table), (f"Latest (`{mine[0]['id']}`):\n\n" + detail) if mine else detail


def admin_list(profile: Optional[gr.OAuthProfile]):
    if not is_organizer(profile):
        return pd.DataFrame()
    rs = sorted(records(fresh=True), key=lambda r: r["created_utc"], reverse=True)
    return pd.DataFrame([
        {"ID": r["id"], "Submitted (UTC)": r["created_utc"][:16], "User": r["hf_user"], "Team": r["team_name"],
         "Method": r["method_name"], "Status": r["status"],
         "Failed checks": ", ".join(sorted({i["check"] for i in r["validation"] if i["level"] == "FAIL"}))}
        for r in rs])


def admin_set(sid: str, status: str, note: str, profile: Optional[gr.OAuthProfile]):
    if not is_organizer(profile):
        return "Organizers only."
    try:
        rec = core.set_status(STORE, records(fresh=True), sid.strip(), status, note)
    except core.SubmissionError as e:
        return f"❌ {e}"
    records(fresh=True)
    return f"`{rec['id']}` ({rec['team_name']}) is now **{status}**."


def on_load(profile: Optional[gr.OAuthProfile]):
    return gr.update(visible=is_organizer(profile))


# ───────────────────────────────── UI ──────────────────────────────────
def limits_text() -> str:
    parts = []
    if CFG.limit_per_day is not None:
        parts.append(f"{CFG.limit_per_day} per team per UTC day")
    if CFG.limit_total is not None:
        parts.append(f"{CFG.limit_total} per team in total")
    deadline = f" Deadline: **{CFG.deadline.strftime('%Y-%m-%d %H:%M UTC')}**." if CFG.deadline else ""
    return (f"Limits: {', '.join(parts) if parts else 'none'}; only accepted or pending entries count.{deadline}")


with gr.Blocks(title="RoboEval Challenge · CoRL 2026") as demo:
    gr.Markdown(
        "# RoboEval Diagnostic Bimanual Challenge · CoRL 2026\n"
        f"Track 2 of the CoRL 2026 Embodied AI Benchmark and Sim-to-Real Challenge Suite. "
        f"Release `{spec.CHALLENGE_RELEASE}`, {len(spec.all_variations())} task variations × "
        f"{len(spec.EVAL_SEEDS)} episodes. "
        "[Rules](https://robo-eval.github.io/challenge-rules.html) · "
        "[Submission guidelines](https://robo-eval.github.io/challenge-submission.html)"
    )
    with gr.Tab("Leaderboard"):
        gr.Markdown("Each team's most recent accepted submission, ranked by Success, then Task Progression, "
                    "then submission time, until the composite score is published. ↑ higher is better, "
                    "↓ lower is better. Every entry runs at 20 Hz; **Action** is the team's declared action mode.")
        with gr.Row():
            task = gr.Dropdown(TASK_CHOICES, value="overall", label="Task")
            family = gr.Radio(FAMILY_CHOICES, value="All metrics", label="Metrics")
        with gr.Row():
            mode = gr.Radio(["all", "vision", "state"], value="all", label="Observations")
            action = gr.Dropdown(ACTION_CHOICES, value="All", label="Action mode")
        board = gr.Dataframe(interactive=False, wrap=True)
        refresh = gr.Button("Refresh", size="sm")
    with gr.Tab("Submit"):
        gr.Markdown(
            "1. Build the archive: `python -m roboeval.challenge.package submission/`\n"
            "2. Host it where it downloads without signing in (e.g. a public Hugging Face repo), or upload it here.\n"
            "3. Paste the link and the SHA-256 that `package` printed.\n\n"
            "Validation re-derives your results from the logs and replays "
            f"{CFG.replay_episodes} logged episodes in the simulator; it takes a few minutes. {limits_text()}"
        )
        gr.LoginButton()
        url = gr.Textbox(label="Archive link (https://…/<team>_<method>.tar.gz)")
        upload = gr.File(label="…or upload the archive", file_types=[".gz"], type="filepath")
        sha = gr.Textbox(label="SHA-256 printed by package", max_lines=1)
        agree = gr.Checkbox(label="I have read and followed the challenge rules.")
        go = gr.Button("Submit", variant="primary")
        result = gr.Markdown()
    with gr.Tab("My submissions"):
        mine_btn = gr.Button("Load my submissions", size="sm")
        mine_table = gr.Dataframe(interactive=False)
        mine_detail = gr.Markdown()
    with gr.Tab("Organizers", visible=False) as admin_tab:
        admin_btn = gr.Button("Load all submissions", size="sm")
        admin_table = gr.Dataframe(interactive=False, wrap=True)
        with gr.Row():
            sid = gr.Textbox(label="Submission ID")
            new_status = gr.Dropdown([core.ACCEPTED, core.REVIEW, core.REJECTED, core.HIDDEN],
                                     value=core.ACCEPTED, label="Set status")
            note = gr.Textbox(label="Note (kept in the record)")
        apply = gr.Button("Apply")
        admin_msg = gr.Markdown()

    views = [task, family, mode, action]
    demo.load(show_leaderboard, views, board)
    demo.load(on_load, None, admin_tab)
    for control in views:
        control.change(show_leaderboard, views, board)
    refresh.click(lambda *v: (records(fresh=True), show_leaderboard(*v))[1], views, board)
    # Grey the button out while a submission is in flight so it cannot be pressed twice;
    # .then() runs whether validation passed, failed, or raised.
    go.click(lambda: gr.update(interactive=False, value="Validating… please wait"), None, go, queue=False) \
        .then(submit, [url, upload, sha, agree], result, concurrency_limit=1, concurrency_id="validate") \
        .then(lambda: gr.update(interactive=True, value="Submit"), None, go, queue=False)
    mine_btn.click(my_submissions, None, [mine_table, mine_detail])
    admin_btn.click(admin_list, None, admin_table)
    apply.click(admin_set, [sid, new_status, note], admin_msg)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(server_name="0.0.0.0", server_port=7860,
                                                   max_file_size=f"{int(CFG.max_archive_gb)}gb")
