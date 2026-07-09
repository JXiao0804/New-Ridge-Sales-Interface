import warnings warnings.filterwarnings("ignore")
import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib matplotlib,use("Agg")
import matplotlib.patches as mpatches
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="New Ridge Sales Interface",
    page_icon="📦",
    layout="wide",
)

LEAD_TIME_DAYS     = 14
SAFETY_BUFFER_DAYS = 7
REORDER_POINT_DAYS = LEAD_TIME_DAYS + SAFETY_BUFFER_DAYS

MONTH_COLS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sept","Oct","Nov","Dec"]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]  = ["Times New Roman", "DejaVu Serif"]
TITLE_SIZE = 14
LABEL_SIZE = 11
COLOR_CRIT = "#d62728"
COLOR_WARN = "#ff7f0e"
COLOR_OK   = "#2ca02c"
PALETTE    = {"CRITICAL": COLOR_CRIT, "WARNING": COLOR_WARN,
              "OK": COLOR_OK, "NO_MOVEMENT": "#aaaaaa"}

def load_raw(uploaded_file):
    raw = pd.read_csv(uploaded_file, encoding="utf-8-sig", header=None, dtype=str)
    col_names = ["SKU","Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sept","Oct","Nov","Dec",
                 "TOTAL_SALES","Avg_per_mo","On_Hand","Months_supply","_extra"]
    raw.columns = col_names[:raw.shape[1]]
    raw = raw[raw["SKU"].notna()]
    raw = raw[raw["SKU"].str.strip() != ""]
    return raw.reset_index(drop=True)


def clean_block(df, year_tag):
    out = df.copy()
    out = out[~out["SKU"].str.match(r"^(Sku\s+\d{4}|Yellow|Jan)", na=False, case=False)]
    out["year"] = year_tag
    out["SKU"]  = out["SKU"].str.strip()
    out = out[out["SKU"] != ""]
    month_cols_present = [m for m in MONTH_COLS if m in out.columns]
    for col in month_cols_present + ["TOTAL_SALES","Avg_per_mo","On_Hand","Months_supply"]:
        if col in out.columns:
            out[col] = (
                out[col].astype(str)
                .str.replace(r"[^\d.\-]", "", regex=True)
                .replace("", np.nan)
            )
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["TOTAL_SALES"] = out[month_cols_present].sum(axis=1, min_count=1)
    active             = out[month_cols_present].notna().sum(axis=1).clip(lower=1)
    out["Avg_per_mo"]  = out["TOTAL_SALES"] / active
    return out.reset_index(drop=True)
 
 
def split_and_parse(raw):
    header_mask = raw["SKU"].str.match(r"^Sku\s+\d{4}", na=False)
    header_idxs = raw.index[header_mask].tolist()
    if len(header_idxs) < 2:
        raise ValueError(f"Expected ≥2 year-block headers, found {len(header_idxs)}")
    df_2026 = clean_block(raw.iloc[header_idxs[0]+1 : header_idxs[1]], "2026")
    hist_frames = []
    for i in range(1, len(header_idxs)):
        start    = header_idxs[i]
        end      = header_idxs[i+1] if i+1 < len(header_idxs) else len(raw)
        year_str = re.search(r"\d{4}", raw.iloc[start]["SKU"])
        year_tag = year_str.group() if year_str else f"hist_{i}"
        block    = raw.iloc[start+1 : end].copy()
        hist_frames.append(clean_block(block, year_tag))
    df_hist = pd.concat(hist_frames, ignore_index=True)
    return df_2026, df_hist
 
 
def build_features(df_2026, df_hist):
    hist_avg = (df_hist.groupby("SKU")["Avg_per_mo"]
                .mean().rename("hist_avg_per_mo").reset_index())
    df_2025  = df_hist[df_hist["year"] == "2025"]
    avg_2025 = (df_2025.groupby("SKU")["Avg_per_mo"]
                .mean().rename("avg_2025").reset_index())
    ytd_months = [m for m in MONTH_COLS if m in df_2026.columns
                  and df_2026[m].notna().any()]
    base = df_2026[["SKU","Avg_per_mo","On_Hand"] + ytd_months].copy()
    base.rename(columns={"Avg_per_mo": "ytd_avg_per_mo"}, inplace=True)
    master = (base.merge(hist_avg, on="SKU", how="left")
                  .merge(avg_2025,  on="SKU", how="left"))
    master["sell_rate"]  = (master["avg_2025"]
                            .fillna(master["hist_avg_per_mo"])
                            .fillna(master["ytd_avg_per_mo"]))
    master["daily_rate"] = master["sell_rate"] / 30.44
    master["on_hand"]    = pd.to_numeric(master["On_Hand"], errors="coerce").fillna(0)
    master["days_remaining"] = np.where(
        master["daily_rate"] > 0,
        master["on_hand"] / master["daily_rate"],
        np.inf
    )
    def urgency(row):
        dr = row["days_remaining"]
        if dr == np.inf:          return "NO_MOVEMENT"
        if row["on_hand"] == 0 or dr <= LEAD_TIME_DAYS: return "CRITICAL"
        if dr <= REORDER_POINT_DAYS: return "WARNING"
        return "OK"
    master["urgency"]             = master.apply(urgency, axis=1)
    master["suggested_order_qty"] = (master["sell_rate"] * 3).round(0).astype("Int64")
    master["est_stockout_days"]   = master["days_remaining"].replace(np.inf, np.nan)
    return master
 
 
def forecast_skus(df_hist, n_months_ahead=6):
    records = []
    for sku, grp in df_hist.groupby("SKU"):
        vals = []
        for _, row in grp.iterrows():
            for m in MONTH_COLS:
                if m in row and pd.notna(row[m]):
                    vals.append(float(row[m]))
        if len(vals) < 6:
            continue
        X = np.arange(len(vals)).reshape(-1, 1)
        y = np.array(vals)
        lr = LinearRegression().fit(X, y)
        Xf = np.arange(len(vals), len(vals)+n_months_ahead).reshape(-1, 1)
        lr_preds = lr.predict(Xf).clip(min=0)
        if len(vals) >= 12:
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            forecast = (lr_preds + rf.predict(Xf).clip(min=0)) / 2
        else:
            forecast = lr_preds
        records.append({
            "SKU":             sku,
            "forecast_avg_mo": round(float(np.mean(forecast)), 1),
            "trend":           "up" if lr.coef_[0] > 0.5 else ("down" if lr.coef_[0] < -0.5 else "flat"),
        })
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["SKU","forecast_avg_mo","trend"])