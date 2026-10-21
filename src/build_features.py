"""
Build one row per NFL game with features that use ONLY information available before kickoff.
Saves to the `features` table in data/nfl.duckdb.

Run from the repo root:  python src\\build_features.py
"""
import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge

DB_PATH = "data/nfl.duckdb"
PRIOR_GAMES = 6          # games of evidence before this season's data weighs as much as last season's
RIDGE_ALPHA = 3.0        # shrinkage for team ratings
MARGIN_CAP = 24          # cap blowouts in the power rating
QB_WINDOW = 32           # QB value uses his last 32 games played
QB_SHRINK = 250          # dropbacks of evidence before a QB's own numbers outweigh the prior
QB_PRIOR = -0.05         # EPA/dropback assumed for a QB with no NFL history (backup level)
RELOCATED = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
RATING_COLS = ["margin_rtg", "off_epa", "def_epa", "off_sr", "def_sr"]


# ---------------------------------------------------------------- load
def load(con):
    g = con.execute("SELECT * FROM games").df()
    tg = con.execute("SELECT * FROM team_games").df()
    qb = con.execute("SELECT * FROM qb_games").df()
    for df, cols in ((g, ["home_team", "away_team"]), (tg, ["team", "opponent"]), (qb, ["team"])):
        for c in cols:
            df[c] = df[c].replace(RELOCATED)
    g["kickoff"] = pd.to_datetime(g["gameday"] + " " + g["gametime"].fillna("13:00"), errors="coerce")
    g["postseason"] = (g["game_type"] != "REG").astype(int)
    g["period"] = g["week"]
    g["completed"] = g["home_score"].notna() & g["away_score"].notna()
    g["margin"] = g["home_score"] - g["away_score"]
    g["neutral"] = (g["location"] == "Neutral").astype(int)
    return g.sort_values(["season", "period", "kickoff"]).reset_index(drop=True), tg, qb


# ---------------------------------------------------------------- team ratings
def team_rows(g, tg):
    done = g[g["completed"]]
    loc = 1 - done["neutral"]
    home = pd.DataFrame({"game_id": done["game_id"], "season": done["season"], "period": done["period"],
                         "team": done["home_team"], "opponent": done["away_team"], "loc": loc,
                         "pts_margin": done["margin"]})
    away = pd.DataFrame({"game_id": done["game_id"], "season": done["season"], "period": done["period"],
                         "team": done["away_team"], "opponent": done["home_team"], "loc": -loc,
                         "pts_margin": -done["margin"]})
    rows = pd.concat([home, away], ignore_index=True)
    stats = tg[["game_id", "team", "epa_play", "success_rate"]]
    return rows.merge(stats, on=["game_id", "team"], how="left")


def fit_margin(d, teams):
    idx = {t: i for i, t in enumerate(teams)}
    n, T = len(d), len(teams)
    r = np.arange(n)
    X = csr_matrix((np.r_[np.ones(n), -np.ones(n)],
                    (np.r_[r, r], np.r_[d["team"].map(idx), d["opponent"].map(idx)])), shape=(n, T))
    X = hstack([X, csr_matrix(d[["loc"]].values.astype(float))]).tocsr()
    m = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False).fit(X, d["pts_margin"].clip(-MARGIN_CAP, MARGIN_CAP))
    return dict(zip(teams, m.coef_[:T]))


def fit_off_def(d, metric, teams):
    d = d.dropna(subset=[metric])
    idx = {t: i for i, t in enumerate(teams)}
    n, T = len(d), len(teams)
    r = np.arange(n)
    X = csr_matrix((np.ones(2 * n), (np.r_[r, r], np.r_[d["team"].map(idx), T + d["opponent"].map(idx)])),
                   shape=(n, 2 * T))
    X = hstack([X, csr_matrix(d[["loc"]].values.astype(float))]).tocsr()
    m = Ridge(alpha=RIDGE_ALPHA).fit(X, d[metric])
    return dict(zip(teams, m.coef_[:T])), dict(zip(teams, m.coef_[T:2 * T]))


def ratings_from(d, teams):
    out = pd.DataFrame({"team": teams})
    gp = d.groupby("team").size()
    out["games_played"] = out["team"].map(gp).fillna(0)
    if len(d) < 16:
        for c in RATING_COLS:
            out[c] = 0.0
        return out
    out["margin_rtg"] = out["team"].map(fit_margin(d, teams))
    o, df_ = fit_off_def(d, "epa_play", teams)
    out["off_epa"], out["def_epa"] = out["team"].map(o), out["team"].map(df_)
    o, df_ = fit_off_def(d, "success_rate", teams)
    out["off_sr"], out["def_sr"] = out["team"].map(o), out["team"].map(df_)
    return out


def build_ratings(g, rows):
    weekly, final = [], []
    for season, sg in g.groupby("season"):
        teams = sorted(set(sg["home_team"]) | set(sg["away_team"]))
        sr = rows[rows["season"] == season]
        for period in sorted(sg["period"].unique()):
            r = ratings_from(sr[sr["period"] < period], teams)
            r["season"], r["period"] = season, period
            weekly.append(r)
        f = ratings_from(sr, teams)
        f["season"] = season
        final.append(f)
    print(f"  team ratings built for {g['season'].min()}-{g['season'].max()}")
    return pd.concat(weekly, ignore_index=True), pd.concat(final, ignore_index=True)


# ---------------------------------------------------------------- quarterback value
def qb_values(g, qb):
    """Pre-game value (shrunk EPA/dropback over his last QB_WINDOW games) for each listed starter."""
    kick = g[["game_id", "kickoff"]]
    q = qb.merge(kick, on="game_id").sort_values(["qb_id", "kickoff"])
    grp = q.groupby("qb_id")
    # stats from games strictly BEFORE this one
    q["db_prior"] = grp["dropbacks"].transform(lambda s: s.shift().rolling(QB_WINDOW, min_periods=1).sum()).fillna(0)
    q["epa_prior"] = grp["epa_sum"].transform(lambda s: s.shift().rolling(QB_WINDOW, min_periods=1).sum()).fillna(0)
    # career snapshot after each game, used to value him in his NEXT game (incl. future games)
    q["db_after"] = grp["dropbacks"].transform(lambda s: s.rolling(QB_WINDOW, min_periods=1).sum())
    q["epa_after"] = grp["epa_sum"].transform(lambda s: s.rolling(QB_WINDOW, min_periods=1).sum())
    shrink = lambda e, d: (e + QB_SHRINK * QB_PRIOR) / (d + QB_SHRINK)

    out = []
    for side in ("home", "away"):
        s = g[["game_id", "kickoff", f"{side}_qb_id"]].rename(columns={f"{side}_qb_id": "qb_id"})
        s = s.dropna(subset=["qb_id"]).sort_values("kickoff")
        snap = q[["qb_id", "kickoff", "db_after", "epa_after"]].sort_values("kickoff")
        # latest completed game for that QB strictly before this kickoff
        m = pd.merge_asof(s, snap.rename(columns={"kickoff": "prev_kick"}), left_on="kickoff",
                          right_on="prev_kick", by="qb_id", allow_exact_matches=False, direction="backward")
        m["value"] = shrink(m["epa_after"].fillna(0), m["db_after"].fillna(0))
        m["dropbacks_hist"] = m["db_after"].fillna(0)
        out.append(m[["game_id", "value", "dropbacks_hist"]].rename(
            columns={"value": f"{side}_qb_value", "dropbacks_hist": f"{side}_qb_hist"}))
    return out[0].merge(out[1], on="game_id", how="outer")


def qb_change(g, vals):
    """How much better/worse this week's starter is than the QB the team started last game."""
    v = g[["game_id", "season", "kickoff", "home_team", "away_team"]].merge(vals, on="game_id", how="left")
    long = pd.concat([
        v[["game_id", "season", "kickoff", "home_team", "home_qb_value"]].set_axis(["game_id", "season", "kickoff", "team", "qbv"], axis=1),
        v[["game_id", "season", "kickoff", "away_team", "away_qb_value"]].set_axis(["game_id", "season", "kickoff", "team", "qbv"], axis=1),
    ]).sort_values(["team", "kickoff"])
    long["prev_qbv"] = long.groupby(["team", "season"])["qbv"].shift()
    long["qb_delta"] = (long["qbv"] - long["prev_qbv"]).fillna(0)
    return long[["game_id", "team", "qb_delta"]]


# ---------------------------------------------------------------- assemble
def side(df, prefix, team_col, keys):
    return df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c not in keys + ["team"]}).rename(columns={"team": team_col})


def assemble(g, weekly, final, vals, deltas):
    f = g.copy()
    f = f.merge(side(weekly, "h", "home_team", ["season", "period"]), on=["season", "period", "home_team"], how="left")
    f = f.merge(side(weekly, "a", "away_team", ["season", "period"]), on=["season", "period", "away_team"], how="left")
    prev = final[["season", "team"] + RATING_COLS].copy()
    prev["season"] += 1
    prev = prev.rename(columns={c: f"prev_{c}" for c in RATING_COLS})
    f = f.merge(side(prev, "h", "home_team", ["season"]), on=["season", "home_team"], how="left")
    f = f.merge(side(prev, "a", "away_team", ["season"]), on=["season", "away_team"], how="left")
    f = f.merge(vals, on="game_id", how="left")
    f = f.merge(deltas.rename(columns={"team": "home_team", "qb_delta": "h_qb_delta"}), on=["game_id", "home_team"], how="left")
    f = f.merge(deltas.rename(columns={"team": "away_team", "qb_delta": "a_qb_delta"}), on=["game_id", "away_team"], how="left")

    for c in [c for c in f.columns if c[:2] in ("h_", "a_") and any(c.endswith(r) for r in RATING_COLS)]:
        f[c] = f[c].fillna(0.0)
    for c in ("h_games_played", "a_games_played", "h_qb_delta", "a_qb_delta"):
        f[c] = f[c].fillna(0)
    for c in ("home_qb_value", "away_qb_value"):
        f[c] = f[c].fillna(QB_PRIOR)

    gp = (f["h_games_played"] + f["a_games_played"]) / 2
    f["w_inseason"] = gp / (gp + PRIOR_GAMES)
    w = f["w_inseason"]

    f["in_margin"] = f["h_margin_rtg"] - f["a_margin_rtg"]
    f["in_epa_edge"] = (f["h_off_epa"] + f["a_def_epa"]) - (f["a_off_epa"] + f["h_def_epa"])
    f["in_sr_edge"] = (f["h_off_sr"] + f["a_def_sr"]) - (f["a_off_sr"] + f["h_def_sr"])
    f["pr_margin"] = f["h_prev_margin_rtg"] - f["a_prev_margin_rtg"]
    f["pr_epa_edge"] = (f["h_prev_off_epa"] + f["a_prev_def_epa"]) - (f["a_prev_off_epa"] + f["h_prev_def_epa"])
    for c in ("in_margin", "in_epa_edge", "in_sr_edge"):
        f[f"{c}_w"] = f[c] * w
    for c in ("pr_margin", "pr_epa_edge"):
        f[f"{c}_w"] = f[c] * (1 - w)
    f["qb_edge"] = f["home_qb_value"] - f["away_qb_value"]
    f["qb_change"] = f["h_qb_delta"] - f["a_qb_delta"]
    f["home_field"] = 1 - f["neutral"]
    f["rest_diff"] = (f["home_rest"] - f["away_rest"]).clip(-7, 7).fillna(0)
    f["div_game"] = f["div_game"].fillna(0).astype(int)
    return f


KEEP = ["game_id", "season", "week", "period", "game_type", "postseason", "kickoff", "home_team", "away_team",
        "home_qb_name", "away_qb_name", "completed", "home_score", "away_score", "margin",
        "w_inseason", "h_games_played", "a_games_played",
        "in_margin", "in_epa_edge", "in_sr_edge", "pr_margin", "pr_epa_edge",
        "in_margin_w", "in_epa_edge_w", "in_sr_edge_w", "pr_margin_w", "pr_epa_edge_w",
        "home_qb_value", "away_qb_value", "home_qb_hist", "away_qb_hist", "qb_edge", "qb_change",
        "home_field", "rest_diff", "div_game", "spread_line"]


def report(f):
    print("\n================ FEATURE TABLE ================")
    print(f.groupby("season").agg(games=("game_id", "size"), completed=("completed", "sum")).T.to_string())
    d = f[f["completed"] & (f["season"].between(2010, 2025))]
    late = d[d["w_inseason"] > 0.4]
    print("\nCorrelation with final margin (2010-2025):")
    for c, sub in (("in_margin", late), ("in_epa_edge", late), ("pr_margin", d), ("qb_edge", d), ("qb_change", d)):
        print(f"  {c:<12} {sub[c].corr(sub['margin']):+.3f}")
    m = d.dropna(subset=["spread_line"])
    print(f"  {'market':<12} {m['spread_line'].corr(m['margin']):+.3f}   (should be clearly positive)")


def main():
    con = duckdb.connect(DB_PATH)
    g, tg, qb = load(con)
    print("Building ratings (only past games each week)...")
    weekly, final = build_ratings(g, team_rows(g, tg))
    vals = qb_values(g, qb)
    f = assemble(g, weekly, final, vals, qb_change(g, vals))[KEEP]
    con.register("f_df", f)
    con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM f_df")
    con.unregister("f_df")
    con.close()
    report(f)
    print(f"\nSaved `features`: {len(f):,} rows")


if __name__ == "__main__":
    main()
