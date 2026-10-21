"""
Grade a week of saved NFL predictions against final scores.
Run from the repo root after Monday Night Football:  python src\\grade_week.py --week 7
"""
import argparse
import os
import sys

import nflreadpy as nfl
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pull_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--season", type=int, default=pull_data.current_season())
    args = ap.parse_args()

    path = os.path.join("predictions", f"{args.season}_week{args.week:02d}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Can't find {path}")
    pred = pd.read_csv(path)
    res = nfl.load_schedules([args.season]).to_pandas()[["game_id", "away_score", "home_score"]]
    d = pred.merge(res, on="game_id", how="left").dropna(subset=["home_score", "away_score"])
    d = d[d["home_score"] != d["away_score"]].copy()
    if d.empty:
        raise SystemExit("No finished games for that week yet.")
    d["winner"] = d["home"].where(d["home_score"] > d["away_score"], d["away"])
    d["correct"] = d["pick"] == d["winner"]
    d["score"] = d["away"] + " " + d["away_score"].astype(int).astype(str) + ", " + d["home"] + " " + d["home_score"].astype(int).astype(str)
    m = d.dropna(subset=["market_spread"])
    m = m[m["market_spread"] != 0]
    mkt_right = int(((m["market_spread"] < 0) == (m["home_score"] > m["away_score"])).sum())

    n, c = len(d), int(d["correct"].sum())
    print("\n" + "=" * 64)
    print(f" WEEK {args.week} RESULTS: {c}-{n - c} ({c / n:.1%}) | market favorite {mkt_right}-{len(m) - mkt_right}")
    print("=" * 64)
    for tier in ["HIGH", "MED", "LOW"]:
        t = d[d["conf"] == tier]
        if len(t):
            print(f"  {tier:<5} {int(t['correct'].sum())}-{int((~t['correct']).sum())}  (avg predicted {t['win_pct'].mean():.0f}%)")
    print("\nMISSES")
    for _, r in d[~d["correct"]].sort_values("win_pct", ascending=False).iterrows():
        print(f"  Picked {r['pick']} at {r['win_pct']:.0f}%  ->  {r['score']}")
    q = d[d["qb_note"].fillna("") != ""]
    if len(q):
        print(f"\nQB-CHANGE GAMES: {int(q['correct'].sum())}-{int((~q['correct']).sum())}")
    print()


if __name__ == "__main__":
    main()
