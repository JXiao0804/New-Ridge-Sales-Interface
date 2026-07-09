import warnings
warnings.filterwarnings("ignore")
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

CSV_PATH  = "Sales by Month.csv"
LEAD_TIME_DAYS      = 14
SAFETY_BUFFER_DAYS  = 7
REORDER_POINT_DAYS  = LEAD_TIME_DAYS + SAFETY_BUFFER_DAYS   # 21 days total
OUTPUT_PNG          = "dashboard.png"

MONTH_COLS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sept","Oct","Nov","Dec"]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]  = ["Times New Roman", "DejaVu Serif"]
TITLE_SIZE  = 14
LABEL_SIZE  = 11
COLOR_CRIT  = "#d62728"
COLOR_WARN  = "#ff7f0e"
COLOR_OK    = "#2ca02c"



def load_raw(path):
    raw = pd.read_csv(path, encoding="utf-8-sig", header=None, dtype=str)
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
    """
    The CSV has header rows like 'Sku 2026', 'Sku 2025', etc.
    Split on those and return (df_2026, df_historical).
    """
    header_mask = raw["SKU"].str.match(r"^Sku\s+\d{4}", na=False)
    header_idxs = raw.index[header_mask].tolist()

    if len(header_idxs) < 2:
        raise ValueError(f"Expected ≥2 year-block headers, found {len(header_idxs)}: {raw.iloc[header_idxs]['SKU'].tolist()}")

    df_2026 = clean_block(raw.iloc[header_idxs[0]+1 : header_idxs[1]], "2026")

    # Historical blocks: 2025 onward
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
    hist_avg = (
        df_hist.groupby("SKU")["Avg_per_mo"]
        .mean().rename("hist_avg_per_mo").reset_index()
    )
    df_2025  = df_hist[df_hist["year"] == "2025"]
    avg_2025 = (
        df_2025.groupby("SKU")["Avg_per_mo"]
        .mean().rename("avg_2025").reset_index()
    )

    ytd_months = [m for m in MONTH_COLS if m in df_2026.columns
                  and df_2026[m].notna().any()]

    base = df_2026[["SKU","Avg_per_mo","On_Hand"] + ytd_months].copy()
    base.rename(columns={"Avg_per_mo": "ytd_avg_per_mo"}, inplace=True)

    master = (
        base
        .merge(hist_avg, on="SKU", how="left")
        .merge(avg_2025,  on="SKU", how="left")
    )

    master["sell_rate"]  = (
        master["avg_2025"]
        .fillna(master["hist_avg_per_mo"])
        .fillna(master["ytd_avg_per_mo"])
    )
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

    master["urgency"]              = master.apply(urgency, axis=1)
    master["suggested_order_qty"]  = (master["sell_rate"] * 3).round(0).astype("Int64")
    master["est_stockout_days"]    = master["days_remaining"].replace(np.inf, np.nan)

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
            forecast = ((lr_preds + rf.predict(Xf).clip(min=0)) / 2)
        else:
            forecast = lr_preds

        records.append({
            "SKU":             sku,
            "forecast_avg_mo": round(float(np.mean(forecast)), 1),
            "trend":           "up" if lr.coef_[0] > 0.5 else ("down" if lr.coef_[0] < -0.5 else "flat"),
        })
    return pd.DataFrame(records, columns=["SKU", "forecast_avg_mo", "trend"]) if records else pd.DataFrame(columns=["SKU", "forecast_avg_mo", "trend"])



def build_dashboard(master, forecast_df, output_path):
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#f9f9f7")
    fig.suptitle("New Ridge Sales Interface — Inventory & Reorder Dashboard",
                 fontsize=18, fontweight="bold", y=0.97)

    axes = fig.subplot_mosaic(
        [["urgency_bar",  "top_skus"],
         ["days_scatter", "stockout_hist"],
         ["forecast_bar", "trend_pie"]]
    )

    palette = {"CRITICAL": COLOR_CRIT, "WARNING": COLOR_WARN,
               "OK": COLOR_OK, "NO_MOVEMENT": "#aaaaaa"}

    ax   = axes["urgency_bar"]
    order   = ["CRITICAL","WARNING","OK","NO_MOVEMENT"]
    counts  = master["urgency"].value_counts().reindex(order, fill_value=0)
    bars    = ax.bar(counts.index, counts.values,
                     color=[palette[k] for k in order], edgecolor="white", linewidth=1.4)
    ax.set_title("SKU Count by Reorder Urgency", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_ylabel("# SKUs", fontsize=LABEL_SIZE)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_facecolor("#f9f9f7")

    ax = axes["top_skus"]
    urgent = (
        master[master["urgency"].isin(["CRITICAL","WARNING"])]
        .dropna(subset=["est_stockout_days"])
        .nsmallest(15, "est_stockout_days")
    )
    if not urgent.empty:
        ax.barh(urgent["SKU"].str[:24], urgent["est_stockout_days"],
                color=[palette[u] for u in urgent["urgency"]], edgecolor="white")
        ax.axvline(REORDER_POINT_DAYS, color="gray",  linestyle="--", linewidth=1.2,
                   label=f"Reorder point ({REORDER_POINT_DAYS}d)")
        ax.axvline(LEAD_TIME_DAYS,     color=COLOR_CRIT, linestyle=":", linewidth=1.2,
                   label=f"Lead time ({LEAD_TIME_DAYS}d)")
        ax.set_xlabel("Days of Stock Remaining", fontsize=LABEL_SIZE)
        ax.legend(fontsize=9); ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No urgent SKUs", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
    ax.set_title("Top 15 Most Urgent SKUs", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_facecolor("#f9f9f7")

    ax = axes["days_scatter"]
    plot_df = master[master["days_remaining"] < 365]
    ax.scatter(plot_df["sell_rate"], plot_df["days_remaining"],
               c=[palette[u] for u in plot_df["urgency"]],
               alpha=0.65, s=40, edgecolors="white", linewidths=0.4)
    ax.axhline(REORDER_POINT_DAYS, color="gray",     linestyle="--", linewidth=1,
               label=f"Reorder point ({REORDER_POINT_DAYS}d)")
    ax.axhline(LEAD_TIME_DAYS,     color=COLOR_CRIT, linestyle=":",  linewidth=1,
               label=f"Lead time ({LEAD_TIME_DAYS}d)")
    ax.set_xlabel("Avg Monthly Sell Rate", fontsize=LABEL_SIZE)
    ax.set_ylabel("Days of Stock Remaining", fontsize=LABEL_SIZE)
    ax.set_title("Sell Rate vs Days Remaining", fontsize=TITLE_SIZE, fontweight="bold")
    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in palette.items()]
    ax.legend(handles=legend_patches, fontsize=8, ncol=2)
    ax.set_facecolor("#f9f9f7")

    ax = axes["stockout_hist"]
    finite = master["est_stockout_days"].dropna()
    finite = finite[finite < 365]
    if not finite.empty:
        ax.hist(finite, bins=30, color="#4c72b0", edgecolor="white", linewidth=0.8)
        ax.axvline(REORDER_POINT_DAYS, color="gray",     linestyle="--",
                   label=f"Reorder point ({REORDER_POINT_DAYS}d)")
        ax.axvline(LEAD_TIME_DAYS,     color=COLOR_CRIT, linestyle=":",
                   label=f"Lead time ({LEAD_TIME_DAYS}d)")
        ax.legend(fontsize=9)
    ax.set_xlabel("Days of Stock Remaining", fontsize=LABEL_SIZE)
    ax.set_ylabel("# SKUs", fontsize=LABEL_SIZE)
    ax.set_title("Distribution of Stock Runway", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_facecolor("#f9f9f7")

    ax = axes["forecast_bar"]
    if not forecast_df.empty:
        top_fc = forecast_df.nlargest(15, "forecast_avg_mo")
        ax.barh(top_fc["SKU"].str[:24], top_fc["forecast_avg_mo"],
                color="#4c72b0", edgecolor="white")
        ax.set_xlabel("Forecasted Avg Monthly Sales (next 6 mo)", fontsize=LABEL_SIZE)
        ax.set_title("Top 15 Forecasted High-Volume SKUs", fontsize=TITLE_SIZE, fontweight="bold")
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                fontsize=12, transform=ax.transAxes)
        ax.set_title("Sales Forecast", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_facecolor("#f9f9f7")

    ax = axes["trend_pie"]
    if not forecast_df.empty:
        tc_map   = {"up": COLOR_OK, "flat": COLOR_WARN, "down": COLOR_CRIT}
        tvc      = forecast_df["trend"].value_counts()
        ax.pie(tvc.values, labels=tvc.index,
               colors=[tc_map.get(t,"#aaa") for t in tvc.index],
               autopct="%1.0f%%", startangle=140,
               textprops={"fontsize": 11})
        ax.set_title("Sales Trend Direction", fontsize=TITLE_SIZE, fontweight="bold")
    else:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Dashboard saved → {output_path}")



def print_reorder_report(master):
    print("\n" + "="*72)
    print("  REORDER ALERT REPORT")
    print(f"  Lead time: {LEAD_TIME_DAYS}d | Safety buffer: {SAFETY_BUFFER_DAYS}d "
          f"| Reorder point: {REORDER_POINT_DAYS}d")
    print("="*72)

    for level in ["CRITICAL","WARNING"]:
        subset = master[master["urgency"] == level].sort_values("days_remaining")
        if subset.empty:
            continue
        label = "🔴 CRITICAL — ORDER NOW" if level == "CRITICAL" else "🟡 WARNING — ORDER SOON"
        print(f"\n{label}  ({len(subset)} SKUs)")
        print(f"  {'SKU':<30} {'On Hand':>8} {'Days Left':>10} {'Sell/mo':>8} {'Suggest Qty':>12}")
        print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8} {'-'*12}")
        for _, row in subset.iterrows():
            dr = f"{row['days_remaining']:.0f}" if row['days_remaining'] != np.inf else "∞"
            sq = int(row["suggested_order_qty"]) if pd.notna(row["suggested_order_qty"]) else 0
            print(f"  {row['SKU']:<30} {int(row['on_hand']):>8} {dr:>10} "
                  f"{row['sell_rate']:>8.1f} {sq:>12}")

    ok = (master["urgency"] == "OK").sum()
    nm = (master["urgency"] == "NO_MOVEMENT").sum()
    print(f"\n🟢 OK: {ok} SKUs | ⚪ NO_MOVEMENT (zero sales rate): {nm} SKUs")
    print("="*72 + "\n")



def main():
    print("New Ridge Sales Interface — Project Runner")
    print("------------------------------------------")

    print("Loading CSV …")
    raw = load_raw(CSV_PATH)

    print("Splitting year blocks …")
    df_2026, df_hist = split_and_parse(raw)
    print(f"  2026 YTD : {len(df_2026)} SKUs")
    print(f"  Historical: {len(df_hist)} rows | years: {sorted(df_hist['year'].unique())}")

    print("Building feature table …")
    master = build_features(df_2026, df_hist)
    print(f"  {len(master)} SKUs | urgency breakdown:")
    print(master["urgency"].value_counts().to_string(index=True))

    print("Forecasting …")
    forecast_df = forecast_skus(df_hist)
    print(f"  {len(forecast_df)} SKUs with sufficient history")

    print_reorder_report(master)

    print("Building dashboard …")
    build_dashboard(master, forecast_df, OUTPUT_PNG)

    master.to_csv("reorder_signals.csv", index=False)
    forecast_df.to_csv("forecasts.csv", index=False)
    print("  reorder_signals.csv → saved")
    print("  forecasts.csv       → saved")
    print("\nDone.")


if __name__ == "__main__":
    main()