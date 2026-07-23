import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        raise ValueError(f"Expected 2+ year-block headers, found {len(header_idxs)}")
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
                  .merge(avg_2025, on="SKU", how="left"))
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
        if dr == np.inf:
            return "NO_MOVEMENT"
        if row["on_hand"] == 0 or dr <= LEAD_TIME_DAYS:
            return "CRITICAL"
        if dr <= REORDER_POINT_DAYS:
            return "WARNING"
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
        Xf = np.arange(len(vals), len(vals) + n_months_ahead).reshape(-1, 1)
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


def fig_urgency_bar(master):
    order  = ["CRITICAL","WARNING","OK","NO_MOVEMENT"]
    counts = master["urgency"].value_counts().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#f9f9f7")
    ax.set_facecolor("#f9f9f7")
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[k] for k in order], edgecolor="white", linewidth=1.4)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("SKU Count by Reorder Urgency", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_ylabel("# SKUs", fontsize=LABEL_SIZE)
    plt.tight_layout()
    return fig


def fig_top_urgent(master):
    urgent = (master[master["urgency"].isin(["CRITICAL","WARNING"])]
              .dropna(subset=["est_stockout_days"])
              .nsmallest(15, "est_stockout_days"))
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#f9f9f7")
    ax.set_facecolor("#f9f9f7")
    if not urgent.empty:
        ax.barh(urgent["SKU"].str[:28], urgent["est_stockout_days"],
                color=[PALETTE[u] for u in urgent["urgency"]], edgecolor="white")
        ax.axvline(REORDER_POINT_DAYS, color="gray", linestyle="--", linewidth=1.2,
                   label=f"Reorder point ({REORDER_POINT_DAYS}d)")
        ax.axvline(LEAD_TIME_DAYS, color=COLOR_CRIT, linestyle=":", linewidth=1.2,
                   label=f"Lead time ({LEAD_TIME_DAYS}d)")
        ax.legend(fontsize=9)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No urgent SKUs", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
    ax.set_title("Top 15 Most Urgent SKUs", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Days of Stock Remaining", fontsize=LABEL_SIZE)
    plt.tight_layout()
    return fig


def fig_scatter(master):
    plot_df = master[master["days_remaining"] < 365]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#f9f9f7")
    ax.set_facecolor("#f9f9f7")
    ax.scatter(plot_df["sell_rate"], plot_df["days_remaining"],
               c=[PALETTE[u] for u in plot_df["urgency"]],
               alpha=0.65, s=40, edgecolors="white", linewidths=0.4)
    ax.axhline(REORDER_POINT_DAYS, color="gray", linestyle="--", linewidth=1,
               label=f"Reorder point ({REORDER_POINT_DAYS}d)")
    ax.axhline(LEAD_TIME_DAYS, color=COLOR_CRIT, linestyle=":", linewidth=1,
               label=f"Lead time ({LEAD_TIME_DAYS}d)")
    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
    ax.legend(handles=legend_patches, fontsize=8, ncol=2)
    ax.set_title("Sell Rate vs Days Remaining", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Avg Monthly Sell Rate", fontsize=LABEL_SIZE)
    ax.set_ylabel("Days of Stock Remaining", fontsize=LABEL_SIZE)
    plt.tight_layout()
    return fig


def fig_stockout_hist(master):
    finite = master["est_stockout_days"].dropna()
    finite = finite[finite < 365]
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#f9f9f7")
    ax.set_facecolor("#f9f9f7")
    if not finite.empty:
        ax.hist(finite, bins=30, color="#4c72b0", edgecolor="white", linewidth=0.8)
        ax.axvline(REORDER_POINT_DAYS, color="gray", linestyle="--",
                   label=f"Reorder point ({REORDER_POINT_DAYS}d)")
        ax.axvline(LEAD_TIME_DAYS, color=COLOR_CRIT, linestyle=":",
                   label=f"Lead time ({LEAD_TIME_DAYS}d)")
        ax.legend(fontsize=9)
    ax.set_title("Distribution of Stock Runway", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Days of Stock Remaining", fontsize=LABEL_SIZE)
    ax.set_ylabel("# SKUs", fontsize=LABEL_SIZE)
    plt.tight_layout()
    return fig


def fig_forecast_bar(forecast_df):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#f9f9f7")
    ax.set_facecolor("#f9f9f7")
    if not forecast_df.empty:
        top_fc = forecast_df.nlargest(15, "forecast_avg_mo")
        ax.barh(top_fc["SKU"].str[:28], top_fc["forecast_avg_mo"],
                color="#4c72b0", edgecolor="white")
        ax.set_xlabel("Forecasted Avg Monthly Sales (next 6 mo)", fontsize=LABEL_SIZE)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "Insufficient history to forecast", ha="center", va="center",
                fontsize=12, transform=ax.transAxes)
    ax.set_title("Top 15 Forecasted High-Volume SKUs", fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout()
    return fig


def fig_trend_pie(forecast_df):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#f9f9f7")
    if not forecast_df.empty:
        tc_map = {"up": COLOR_OK, "flat": COLOR_WARN, "down": COLOR_CRIT}
        tvc    = forecast_df["trend"].value_counts()
        ax.pie(tvc.values, labels=tvc.index,
               colors=[tc_map.get(t, "#aaa") for t in tvc.index],
               autopct="%1.0f%%", startangle=140,
               textprops={"fontsize": 11})
        ax.set_title("Sales Trend Direction", fontsize=TITLE_SIZE, fontweight="bold")
    else:
        ax.axis("off")
    plt.tight_layout()
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("📦 New Ridge Sales Interface")
st.caption("Upload your monthly sales CSV to generate reorder alerts, forecasts, and inventory charts.")

uploaded = st.file_uploader("Upload Sales by Month CSV", type=["csv"])

if uploaded is None:
    st.info("👆 Upload your CSV file above to get started.")
    st.stop()

with st.spinner("Processing data…"):
    try:
        raw             = load_raw(uploaded)
        df_2026, df_hist = split_and_parse(raw)
        master          = build_features(df_2026, df_hist)
        forecast_df     = forecast_skus(df_hist)
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()

st.subheader("Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total SKUs", len(master))
c2.metric("🔴 Critical", int((master["urgency"] == "CRITICAL").sum()))
c3.metric("🟡 Warning",  int((master["urgency"] == "WARNING").sum()))
c4.metric("🟢 OK",       int((master["urgency"] == "OK").sum()))

st.divider()

st.subheader("🔴 Reorder Alerts")

urgent_df = master[master["urgency"].isin(["CRITICAL","WARNING"])].copy()
urgent_df = urgent_df.sort_values(["urgency", "days_remaining"])

display_cols = {
    "SKU":                 "SKU",
    "urgency":             "Status",
    "on_hand":             "On Hand",
    "sell_rate":           "Avg Sales/Mo",
    "est_stockout_days":   "Days of Stock",
    "suggested_order_qty": "Suggested Order Qty",
}

if urgent_df.empty:
    st.success("No SKUs currently need reordering.")
else:
    table = urgent_df[list(display_cols.keys())].rename(columns=display_cols)
    table["On Hand"]       = table["On Hand"].astype(int)
    table["Avg Sales/Mo"]  = table["Avg Sales/Mo"].round(1)
    table["Days of Stock"] = table["Days of Stock"].round(1)

    def color_status(val):
        colors = {"CRITICAL": "background-color:#ffd5d5",
                  "WARNING":  "background-color:#fff3cd"}
        return colors.get(val, "")

    st.dataframe(
        table.style.map(color_status, subset=["Status"]),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button("⬇️ Download Reorder Alerts CSV",
                       urgent_df.to_csv(index=False).encode("utf-8"),
                       "reorder_alerts.csv", "text/csv")

st.divider()

st.subheader("📊 Inventory Dashboard")

col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig_urgency_bar(master))
with col2:
    st.pyplot(fig_top_urgent(master))

col3, col4 = st.columns(2)
with col3:
    st.pyplot(fig_scatter(master))
with col4:
    st.pyplot(fig_stockout_hist(master))

st.divider()

st.subheader("📈 Sales Forecast (Next 6 Months)")

col5, col6 = st.columns([2, 1])
with col5:
    st.pyplot(fig_forecast_bar(forecast_df))
with col6:
    st.pyplot(fig_trend_pie(forecast_df))

if not forecast_df.empty:
    st.download_button("⬇️ Download Forecast CSV",
                       forecast_df.to_csv(index=False).encode("utf-8"),
                       "forecasts.csv", "text/csv")

st.divider()

with st.expander("🔍 Full SKU Data Table"):
    st.dataframe(master[[
        "SKU","urgency","on_hand","sell_rate",
        "days_remaining","suggested_order_qty","est_stockout_days"
    ]].rename(columns={
        "urgency":             "Status",
        "on_hand":             "On Hand",
        "sell_rate":           "Avg Sales/Mo",
        "days_remaining":      "Days Remaining",
        "suggested_order_qty": "Suggested Order Qty",
        "est_stockout_days":   "Est. Stockout (days)",
    }), use_container_width=True, hide_index=True)