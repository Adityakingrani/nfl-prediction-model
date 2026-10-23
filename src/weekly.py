"""
Weekly NFL run: refresh this season, rebuild features, train, and predict the next week.

Run from the repo root (any day after Monday Night Football):
    python src\\weekly.py              # normal weekly run
    python src\\weekly.py --no-pull    # skip the data refresh
    python src\\weekly.py --week 9     # predict a specific week
"""
import argparse
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_features
import pull_data

warnings.filterwarnings("ignore")
DB_PATH = "data/nfl.duckdb"
OUT_DIR = "predictions"
MARGIN_CAP = 28
TEAM_NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers", "CHI": "Bears",
    "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers",
    "LV": "Raiders", "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers",
    "TEN": "Titans", "WAS": "Commanders",
}
FEATURES = ["in_margin_w", "in_epa_edge_w", "in_sr_edge_w", "pr_margin_w", "pr_epa_edge_w",
            "home_field", "rest_diff", "qb_edge", "qb_change"]


def train_model(train):
    m = Ridge(alpha=1.0).fit(train[FEATURES], train["margin"].clip(-MARGIN_CAP, MARGIN_CAP))
    return m, np.std(train["margin"] - m.predict(train[FEATURES]))


def confidence(p):
    p = max(p, 1 - p)
    return "HIGH" if p >= 0.70 else "MED" if p >= 0.58 else "LOW"


def season_record(f, season):
    hist = f[f["completed"] & (f["season"] >= 2010)]
    done = hist[hist["season"] == season]
    mr = mn = kr = kn = 0
    for period in sorted(done["period"].unique()):
        train = hist[(hist["season"] < season) | ((hist["season"] == season) & (hist["period"] < period))]
        test = done[(done["period"] == period) & (done["margin"] != 0)]
        model, _ = train_model(train)
        pred = model.predict(test[FEATURES])
        mr += int(((pred > 0) == (test["margin"] > 0)).sum()); mn += len(test)
        m = test.dropna(subset=["spread_line"]); m = m[m["spread_line"] != 0]
        kr += int(((m["spread_line"] > 0) == (m["margin"] > 0)).sum()); kn += len(m)
    return mr, mn, kr, kn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--week", type=int)
    ap.add_argument("--no-app", action="store_true")
    args = ap.parse_args()

    season = pull_data.current_season()
    if not args.no_pull:
        sys.argv = ["pull_data.py", "--season", str(season)]
        pull_data.main()
    print("\nRebuilding features...")
    build_features.main()

    con = duckdb.connect(DB_PATH, read_only=True)
    f = con.execute("SELECT * FROM features").df()
    con.close()
    f[FEATURES] = f[FEATURES].fillna(0)

    upcoming = f[(f["season"] == season) & ~f["completed"]]
    if upcoming.empty:
        raise SystemExit(f"No upcoming games left in {season}.")
    if args.week:
        wk = upcoming[upcoming["week"] == args.week]
    else:
        future = upcoming[upcoming["kickoff"].isna() | (upcoming["kickoff"] >= pd.Timestamp.now() - pd.Timedelta(hours=6))]
        wk = upcoming[upcoming["week"] == future["week"].min()]
    if wk.empty:
        raise SystemExit("No unplayed games found for that week.")
    week = int(wk["week"].iloc[0])

    model, sigma = train_model(f[f["completed"] & (f["season"] >= 2010)])
    g = wk.copy()
    for col in ("home_team", "away_team"):
        g[col] = g[col].map(TEAM_NAMES).fillna(g[col])
    g["pred_margin"] = model.predict(g[FEATURES])
    g["home_prob"] = norm.cdf(g["pred_margin"] / sigma)
    g["pick"] = np.where(g["pred_margin"] >= 0, g["home_team"], g["away_team"])
    g["win_pct"] = np.maximum(g["home_prob"], 1 - g["home_prob"])
    g["pick_margin"] = g["pred_margin"].abs()
    g["conf"] = g["home_prob"].apply(confidence)
    g["market_spread"] = -g["spread_line"]          # home-team view: negative = home favored
    g["flag"] = g["spread_line"].notna() & (g["spread_line"] != 0) & ((g["spread_line"] > 0) != (g["pred_margin"] > 0))
    g["qb_note"] = np.where(g["qb_change"].abs() > 0.05, "QB change", "")
    g["matchup"] = np.where(g["home_field"] == 1, g["away_team"] + " @ " + g["home_team"], g["away_team"] + " vs " + g["home_team"])
    g = g.sort_values(["kickoff", "matchup"])

    mr, mn, kr, kn = season_record(f, season)
    now = datetime.now().strftime("%a %b %d, %I:%M %p")
    W = 132
    print("\n" + "=" * W)
    print(f" NFL PREDICTIONS - {season} WEEK {week}          generated {now}")
    if mn:
        print(f" Model record this season: {mr}-{mn - mr} ({mr / mn:.1%}) | Market favorite: {kr}-{kn - kr} ({kr / kn:.1%})")
    print("=" * W)
    print(f" {'KICKOFF':<15}{'MATCHUP':<26}{'QBs (away / home)':<36}{'PICK':<12}{'WIN %':>6}{'MARGIN':>8}{'MARKET':>8}  CONF")
    for _, r in g.iterrows():
        kick = r["kickoff"].strftime("%a %I:%M %p") if pd.notna(r["kickoff"]) else "TBD"
        qbs = f"{r['away_qb_name'] or '?'} / {r['home_qb_name'] or '?'}"[:34]
        mkt = f"{r['market_spread']:+.1f}" if pd.notna(r["market_spread"]) else "--"
        tags = ("  !" if r["flag"] else "") + (f"  [{r['qb_note']}]" if r["qb_note"] else "")
        print(f" {kick:<15}{r['matchup']:<26}{qbs:<36}{r['pick']:<12}{r['win_pct']:>6.0%}{r['pick_margin']:>+8.1f}{mkt:>8}  {r['conf']:<4}{tags}")
    print("-" * W)
    c = g["conf"].value_counts()
    print(f" {len(g)} games | HIGH: {c.get('HIGH', 0)} | MEDIUM: {c.get('MED', 0)} | LOW: {c.get('LOW', 0)}")
    print(" MARKET = betting spread from the HOME team's view (negative = home favored), shown for comparison only")
    print(f" ! = model picks a different winner than the market ({int(g['flag'].sum())} games)")
    print(" [QB change] = listed starter differs a lot from the QB the team started last game - confirm before trusting")
    print("=" * W)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{season}_week{week:02d}.csv")
    out = g[["kickoff", "away_team", "home_team", "away_qb_name", "home_qb_name", "pick", "win_pct", "pick_margin",
             "market_spread", "conf", "flag", "qb_note", "home_field", "game_id"]].copy()
    out["kickoff"] = out["kickoff"].dt.strftime("%Y-%m-%d %I:%M %p")
    out["win_pct"] = (out["win_pct"] * 100).round(1)
    out["pick_margin"] = out["pick_margin"].round(1)
    out["neutral"] = out.pop("home_field") == 0
    out = out.rename(columns={"away_team": "away", "home_team": "home", "flag": "disagrees_with_market"})
    # keep the original pre-game predictions for games that have already been played this week,
    # so a mid-week rerun never erases Thursday's pick (needed for grading)
    if os.path.exists(path):
        old = pd.read_csv(path)
        kept = old[~old["game_id"].isin(out["game_id"])]
        if len(kept):
            print(f"Kept {len(kept)} earlier prediction(s) for games already played: "
                  + ", ".join(kept["away"] + " @ " + kept["home"]))
            out = pd.concat([kept, out], ignore_index=True)
            out["_k"] = pd.to_datetime(out["kickoff"], format="%Y-%m-%d %I:%M %p", errors="coerce")
            out = out.sort_values("_k").drop(columns="_k")
    out.to_csv(path, index=False)
    with open(path.replace(".csv", ".json"), "w") as fh:
        json.dump({"season": int(season), "label": f"Week {week}", "generated": now, "model_right": mr,
                   "model_games": mn, "market_right": kr, "market_games": kn}, fh)
    print(f"Saved {path}")

    if not args.no_app:
        print("\nOpening the predictions app... (press Ctrl+C here to stop it)")
        app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "main.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", app])


if __name__ == "__main__":
    main()
