"""
Walk-forward backtest: for every week from 2012 to 2025, train ONLY on games played before
that week, predict that week, and score it. Compares models with and without the QB features.

Run from the repo root:  python src\\backtest.py
"""
import warnings
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
import lightgbm as lgb

warnings.filterwarnings("ignore")
DB_PATH = "data/nfl.duckdb"
TEST_SEASONS = range(2012, 2026)
MARGIN_CAP = 28
HFA = 1.8

TEAM = ["in_margin_w", "in_epa_edge_w", "in_sr_edge_w", "pr_margin_w", "pr_epa_edge_w", "home_field", "rest_diff"]
WITH_QB = TEAM + ["qb_edge", "qb_change"]
LGB = ["in_margin", "in_epa_edge", "in_sr_edge", "pr_margin", "pr_epa_edge", "w_inseason",
       "qb_edge", "qb_change", "home_field", "rest_diff", "div_game"]


def ridge(train, test, cols):
    y = train["margin"].clip(-MARGIN_CAP, MARGIN_CAP)
    m = Ridge(alpha=1.0).fit(train[cols], y)
    return m.predict(test[cols]), np.std(train["margin"] - m.predict(train[cols]))


def fit_predict(train, test):
    out = {}
    base = lambda d: d["in_margin_w"] + d["pr_margin_w"] + HFA * d["home_field"]
    out["baseline"] = (base(test), np.std(train["margin"] - base(train)))
    out["ridge_team"] = ridge(train, test, TEAM)
    out["ridge_qb"] = ridge(train, test, WITH_QB)
    g = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=60,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=7)
    g.fit(train[LGB], train["margin"].clip(-MARGIN_CAP, MARGIN_CAP))
    out["lightgbm"] = (g.predict(test[LGB]), out["ridge_qb"][1])
    m = train.dropna(subset=["spread_line"])
    out["market"] = (test["spread_line"], np.std(m["margin"] - m["spread_line"]))
    return out


def score(d, m):
    d = d.dropna(subset=[f"{m}_pred"])
    d = d[d["margin"] != 0]
    p = d[f"{m}_prob"].clip(1e-4, 1 - 1e-4)
    y = (d["margin"] > 0).astype(int)
    dec = d[d[f"{m}_pred"] != 0]
    return {"games": len(d), "accuracy": ((dec[f"{m}_pred"] > 0) == (dec["margin"] > 0)).mean(),
            "brier": ((p - y) ** 2).mean(), "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(),
            "margin_mae": (d[f"{m}_pred"] - d["margin"]).abs().mean()}


def main():
    con = duckdb.connect(DB_PATH)
    f = con.execute("SELECT * FROM features WHERE completed").df().sort_values(["season", "period"])
    feats = sorted(set(TEAM) | set(WITH_QB) | set(LGB))
    f[feats] = f[feats].fillna(0)
    f = f[f["season"] >= 2010]          # 2009 has no prior-season ratings

    rows = []
    for season in TEST_SEASONS:
        for period in sorted(f.loc[f["season"] == season, "period"].unique()):
            train = f[(f["season"] < season) | ((f["season"] == season) & (f["period"] < period))]
            test = f[(f["season"] == season) & (f["period"] == period)].copy()
            for name, (pred, sig) in fit_predict(train, test).items():
                test[f"{name}_pred"] = np.asarray(pred, dtype=float)
                test[f"{name}_prob"] = norm.cdf(test[f"{name}_pred"] / sig)
            rows.append(test)
        print(f"  backtested {season}")
    bt = pd.concat(rows, ignore_index=True)
    models = ["baseline", "ridge_team", "ridge_qb", "lightgbm", "market"]
    common = bt.dropna(subset=["market_pred"])

    print("\n================ OVERALL (2012-2025, same games for all) ================")
    print(pd.DataFrame({m: score(common, m) for m in models}).T.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nACCURACY BY SEASON")
    print(pd.DataFrame({m: common.groupby("season").apply(lambda d: score(d, m)["accuracy"]) for m in models})
          .to_string(float_format=lambda x: f"{x:.1%}"))

    print("\nGAMES WHERE A TEAM CHANGED QUARTERBACK (|qb_change| > 0.05)")
    qc = common[common["qb_change"].abs() > 0.05]
    print(pd.DataFrame({m: score(qc, m) for m in ["ridge_team", "ridge_qb", "market"]}).T[["games", "accuracy", "margin_mae"]]
          .to_string(float_format=lambda x: f"{x:.3f}"))

    best = min(["baseline", "ridge_team", "ridge_qb", "lightgbm"], key=lambda m: score(common, m)["brier"])
    print(f"\nCALIBRATION — {best}")
    fp = np.maximum(common[f"{best}_prob"], 1 - common[f"{best}_prob"])
    won = np.where(common[f"{best}_prob"] >= 0.5, common["margin"] > 0, common["margin"] < 0)
    cal = pd.DataFrame({"prob": fp, "won": won})
    cal["bucket"] = pd.cut(cal["prob"], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True)
    print(cal.groupby("bucket", observed=True).agg(games=("won", "size"), predicted=("prob", "mean"), actual=("won", "mean"))
          .to_string(float_format=lambda x: f"{x:.1%}"))
    print(f"\nBest model by Brier score: {best}")


if __name__ == "__main__":
    main()
