"""Write the portal's ranked entries as website leaderboard JSON.

    PORTAL_STORE=hf:<namespace>/<dataset> HF_TOKEN=... \
        python export_leaderboard.py --out ../Robo-Eval.github.io/static/data

Writes challenge_leaderboard_global.json and challenge_leaderboard_by_task.json
in the same shapes as the site's leaderboard_global.json / leaderboard_by_task.json.
"""
import argparse
import json
from pathlib import Path

import portal_core as core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    store = core.make_store(core.Config.from_env())
    records = store.records()
    ranked = core.leaderboard(records)
    results = {r["id"]: store.read_json(r["id"], "results.json") for r in ranked}
    global_rows, by_task = core.website_export(records, results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "challenge_leaderboard_global.json").write_text(json.dumps(global_rows, indent=1))
    (args.out / "challenge_leaderboard_by_task.json").write_text(json.dumps(by_task, indent=1))
    print(f"Wrote {len(global_rows)} ranked entries to {args.out}")


if __name__ == "__main__":
    main()
