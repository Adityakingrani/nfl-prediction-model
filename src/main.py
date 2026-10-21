"""
NFL predictions app. Reads the files weekly.py saves in the predictions folder.
Opens automatically at the end of weekly.py, or run:  streamlit run app\\main.py
"""
import glob
import json
import os

import pandas as pd
import streamlit as st

PRED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "predictions")
CONF_ORDER = {"HIGH": 0, "MED": 1, "LOW": 2}
CONF_STYLE = {"HIGH": ("#1f7a4d", "#e3f4ea", "High"), "MED": ("#9a6700", "#fff4d6", "Medium"),
              "LOW": ("#57606a", "#eef0f2", "Low")}

st.set_page_config(page_title="NFL Predictions", page_icon="🏈", layout="wide")
st.markdown("""
<style>
  .kick {color:#6e7781; font-size:0.85rem; margin-bottom:2px;}
  .matchup {font-size:1.05rem; font-weight:600;}
  .qbs {color:#6e7781; font-size:0.82rem; margin-bottom:6px;}
  .badge {display:inline-block; padding:1px 8px; border-radius:10px; font-size:0.75rem; font-weight:600; margin-left:6px;}
  .detail {color:#57606a; font-size:0.88rem; line-height:1.5;}
  .flag {color:#bc4c00; font-size:0.85rem; font-weight:600; margin-top:6px;}
  .qbflag {color:#8250df; font-size:0.85rem; font-weight:600; margin-top:4px;}
</style>""", unsafe_allow_html=True)


def list_weeks():
    weeks = []
    for f in sorted(glob.glob(os.path.join(PRED_DIR, "*.csv")), reverse=True):
        j = f.replace(".csv", ".json")
        meta = json.load(open(j)) if os.path.exists(j) else {}
        season, tag = os.path.basename(f)[:-4].split("_", 1)
        weeks.append({"file": f, "title": f"{season} {meta.get('label', tag)}", "meta": meta})
    return weeks


@st.cache_data
def load(path, mtime):
    d = pd.read_csv(path)
    d["kickoff_dt"] = pd.to_datetime(d["kickoff"], format="%Y-%m-%d %I:%M %p", errors="coerce")
    d["qb_note"] = d["qb_note"].fillna("")
    d["disagrees_with_market"] = d["disagrees_with_market"].fillna(False).astype(bool)
    return d


def market_text(r):
    s = r["market_spread"]
    if pd.isna(s):
        return "No line yet"
    if s == 0:
        return "Pick'em"
    return f"{r['home'] if s < 0 else r['away']} -{abs(s):g}"


def card(r):
    fg, bg, word = CONF_STYLE[r["conf"]]
    kick = r["kickoff_dt"].strftime("%a %I:%M %p").replace(" 0", " ") if pd.notna(r["kickoff_dt"]) else "Time TBD"
    sep = "vs" if r["neutral"] else "@"
    with st.container(border=True):
        st.markdown(f'<div class="kick">{kick}{" · neutral site" if r["neutral"] else ""}</div>'
                    f'<div class="matchup">{r["away"]} {sep} {r["home"]}</div>'
                    f'<div class="qbs">QBs: {r["away_qb_name"]} / {r["home_qb_name"]}</div>'
                    f'<div>Pick: <b>{r["pick"]}</b><span class="badge" style="color:{fg};background:{bg}">{word}</span></div>',
                    unsafe_allow_html=True)
        st.progress(min(max(r["win_pct"] / 100, 0.0), 1.0), text=f"{r['win_pct']:.0f}% to win")
        st.markdown(f'<div class="detail">Model: {r["pick"]} by {r["pick_margin"]:.1f}<br>Market: {market_text(r)}</div>'
                    + ('<div class="flag">⚑ Model disagrees with the market</div>' if r["disagrees_with_market"] else "")
                    + ('<div class="qbflag">◆ Quarterback change. Confirm the starter.</div>' if r["qb_note"] else ""),
                    unsafe_allow_html=True)


weeks = list_weeks()
st.title("🏈 NFL Predictions")
if not weeks:
    st.info("No predictions yet. Run `python src\\weekly.py` first.")
    st.stop()

with st.sidebar:
    st.header("Week")
    choice = st.selectbox("Week", [w["title"] for w in weeks], label_visibility="collapsed")
    week = next(w for w in weeks if w["title"] == choice)
    st.header("Filters")
    team = st.text_input("Team", placeholder="e.g. Lions")
    confs = st.multiselect("Confidence", ["HIGH", "MED", "LOW"], default=["HIGH", "MED", "LOW"],
                           format_func=lambda c: CONF_STYLE[c][2])
    only_flags = st.toggle("Only games where model disagrees with market")
    only_qb = st.toggle("Only games with a QB change")
    sort_by = st.radio("Sort by", ["Kickoff time", "Confidence"], horizontal=True)

df = load(week["file"], os.path.getmtime(week["file"]))
meta = week["meta"]
st.subheader(f"{week['title']} predictions")
if meta.get("generated"):
    st.caption(f"Generated {meta['generated']} · The market line is shown for comparison only and is never used by the model.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Games", len(df))
c2.metric("High confidence", int((df["conf"] == "HIGH").sum()))
c3.metric("QB changes", int((df["qb_note"] != "").sum()))
if meta.get("model_games"):
    mr, mn, kr, kn = meta["model_right"], meta["model_games"], meta["market_right"], meta["market_games"]
    c4.metric("Model record (season)", f"{mr}-{mn - mr}", f"{mr / mn:.1%}", delta_color="off")
    c5.metric("Market favorite (season)", f"{kr}-{kn - kr}", f"{kr / kn:.1%}", delta_color="off")

view = df[df["conf"].isin(confs)]
if team:
    t = team.strip().lower()
    view = view[view["home"].str.lower().str.contains(t, regex=False) | view["away"].str.lower().str.contains(t, regex=False)]
if only_flags:
    view = view[view["disagrees_with_market"]]
if only_qb:
    view = view[view["qb_note"] != ""]
view = (view.assign(_c=view["conf"].map(CONF_ORDER)).sort_values(["_c", "win_pct"], ascending=[True, False])
        if sort_by == "Confidence" else view.sort_values(["kickoff_dt", "away"]))

tab_cards, tab_table = st.tabs(["Cards", "Table"])
with tab_cards:
    if view.empty:
        st.info("No games match these filters.")
    else:
        groups = (view.groupby(view["kickoff_dt"].dt.strftime("%A, %b %d").fillna("Time TBD"), sort=False)
                  if sort_by == "Kickoff time" else [("", view)])
        for day, games in groups:
            if day:
                st.markdown(f"#### {day}")
            cols = st.columns(3)
            for i, (_, r) in enumerate(games.iterrows()):
                with cols[i % 3]:
                    card(r)
with tab_table:
    t = view.assign(Kickoff=view["kickoff_dt"].dt.strftime("%a %I:%M %p"), Matchup=view["away"] + " @ " + view["home"],
                    QBs=view["away_qb_name"] + " / " + view["home_qb_name"], Market=view.apply(market_text, axis=1))
    st.dataframe(t[["Kickoff", "Matchup", "QBs", "pick", "win_pct", "pick_margin", "Market", "conf", "qb_note"]]
                 .rename(columns={"pick": "Pick", "win_pct": "Win %", "pick_margin": "Margin", "conf": "Confidence", "qb_note": "Note"}),
                 hide_index=True, width="stretch",
                 column_config={"Win %": st.column_config.ProgressColumn("Win %", min_value=50, max_value=100, format="%.0f%%"),
                                "Margin": st.column_config.NumberColumn(format="%.1f")})
