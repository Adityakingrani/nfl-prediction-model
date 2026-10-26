"""
Pull NFL data from nflverse (free, no API key) and save it to data/nfl.duckdb.

Tables:
  games       one row per game: teams, scores, rest, venue, projected starting QBs, market line
  team_games  one row per team per game: offensive EPA/play, success rate, pass/rush EPA
  qb_games    one row per QB per game: dropbacks and total EPA

Run from the repo root:
    python src\\pull_data.py                 # full history (first time, ~2-4 minutes)
    python src\\pull_data.py --season 2026   # refresh just one season (weekly)
"""
import argparse
import os
from datetime import date

import duckdb
import nflreadpy as nfl
import pandas as pd
import polars as pl

DB_PATH = "data/nfl.duckdb"
FIRST_SEASON = 2009


def current_season():
    today = date.today()
    return today.year if today.month >= 8 else today.year - 1


def pull_games(seasons):
    s = nfl.load_schedules(seasons).to_pandas()
    keep = ["game_id", "season", "game_type", "week", "gameday", "gametime", "away_team", "home_team",
            "away_score", "home_score", "location", "away_rest", "home_rest", "div_game", "roof",
            "spread_line", "away_qb_id", "home_qb_id", "away_qb_name", "home_qb_name"]
    return s[keep]


def pull_season_plays(season):
    p = nfl.load_pbp([season])
    plays = p.filter(
        pl.col("posteam").is_not_null() & pl.col("epa").is_not_null()
        & pl.col("play_type").is_in(["pass", "run"])
    )
    # team-level: "neutral" plays only (win probability 5-95%) so garbage time doesn't skew ratings
    neutral = plays.filter(pl.col("wp").is_between(0.05, 0.95))
    team = (neutral.group_by(["game_id", "posteam", "defteam"])
            .agg(pl.len().alias("plays"),
                 pl.col("epa").mean().alias("epa_play"),
                 pl.col("success").mean().alias("success_rate"),
                 pl.col("epa").filter(pl.col("qb_dropback") == 1).mean().alias("pass_epa"),
                 pl.col("epa").filter(pl.col("qb_dropback") == 0).mean().alias("rush_epa"))
            .rename({"posteam": "team", "defteam": "opponent"}))
    # QB-level: every dropback (all game states), credited to the passer/scrambler
    qb = (plays.filter((pl.col("qb_dropback") == 1) & pl.col("passer_id").is_not_null())
          .group_by(["game_id", "posteam", "passer_id"])
          .agg(pl.len().alias("dropbacks"), pl.col("epa").sum().alias("epa_sum"),
               pl.col("passer").first().alias("qb_name"))
          .rename({"posteam": "team", "passer_id": "qb_id"}))
    team = team.with_columns(pl.lit(season).alias("season")).to_pandas()
    qb = qb.with_columns(pl.lit(season).alias("season")).to_pandas()
    return team, qb


def pull_availability(season):
    """Current-season roster status and injury report, used to catch injured/benched listed starters."""
    r = nfl.load_rosters_weekly([season]).to_pandas()
    r = r[r["position"] == "QB"][["season", "week", "team", "gsis_id", "full_name", "status"]]
    try:
        i = nfl.load_injuries([season]).to_pandas()
        i = i[i["position"] == "QB"][["season", "week", "team", "gsis_id", "full_name", "report_status"]]
    except Exception:
        i = pd.DataFrame(columns=["season", "week", "team", "gsis_id", "full_name", "report_status"])
    return r, i


def save(con, name, df, season=None):
    con.register("tmp_df", df)
    if season is None or not con.execute(
            f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{name}'").fetchone()[0]:
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_df")
    else:
        con.execute(f"DELETE FROM {name} WHERE season = {season}")
        con.execute(f"INSERT INTO {name} BY NAME SELECT * FROM tmp_df")
    con.unregister("tmp_df")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, help="refresh only this season")
    args = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    con = duckdb.connect(DB_PATH)

    last = current_season()
    seasons = [args.season] if args.season else list(range(FIRST_SEASON, last + 1))

    print(f"Schedules {seasons[0]}-{seasons[-1]}...")
    save(con, "games", pull_games(seasons), args.season)

    teams, qbs = [], []
    for s in seasons:
        t, q = pull_season_plays(s)
        teams.append(t)
        qbs.append(q)
        print(f"  play-by-play {s}: {len(t)} team-games, {len(q)} QB-games")
    save(con, "team_games", pd.concat(teams, ignore_index=True), args.season)
    save(con, "qb_games", pd.concat(qbs, ignore_index=True), args.season)

    r, i = pull_availability(last)
    save(con, "qb_rosters", r)
    save(con, "qb_injuries", i)
    print(f"  QB availability {last}: {len(r)} roster rows, {len(i)} injury rows")

    print("\n================ SUMMARY ================")
    print(con.execute("""
        SELECT season, COUNT(*) AS games, SUM(CASE WHEN home_score IS NOT NULL THEN 1 ELSE 0 END) AS completed
        FROM games GROUP BY season ORDER BY season
    """).df().to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
