# imports
import pandas as  pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Constants
safety_days = 7 #avg days from PO to warehouse receipt
lead_time_days = 14 # wiggle room in case of delays / count errors
reorder_point_days = lead_time_days + safety_days # = 21 days
critical_days = reorder_point_days # order NOW
warning_days = reorder_point_days * 2 # order SOON

month_cols = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sept","Oct","Nov","Dec"]
csv_path = "/Users/jackxiao/Desktop/New Ridge Sales Interface/Sales by Month.csv"

def load_data(path : str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding = "utf-8-sig")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = df.columns.str.strip()
    
    sku_col = [c for c in df.columns if c.lower().startswith("sku")][0]
    df.rename(columns={sku_col: "SKU"}, inplace=True)
    df["SKU"] = df["SKU"].astype(str).str.strip()
    
    df = df[df["SKU"].notna() & (df["SKU"] != "") & (df["SKU"] != "nan")]
    
    for col in month_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    on_hand_col = None
    for candidate in ["On Hand 1/1/25", "On Hand", "On_Hand"]:
        if candidate in df.columns:
            on_hand_col = candidate
            break
    if on_hand_col:
        df["On_Hand"] = pd.to_numeric(df[on_hand_col], errors="coerce").fillna(0)
    else:
        df["On_Hand"] = 0
        
    month_data = df[[c for c in month_cols if c in df.columns]]
    df = df[month_data.notna().any(axis=1)].reset_index(drop=True)
 
    return df

def engineering_features(df : pd.DataFrame) -> pd.DataFrame:
    available_months = [c for c in month_cols if c in df.columns]
    
    df["months_with_data"] = df[available_months].notna().sum(axis = 1)
    df["total_sales"] = df[available_months].sum(axis = 1, skipna= True)
    df["avg_monthly_rate"] = df["total_sales"] / df["months_with_data"].replace(0, np.nan)
    df["avg_daily_rate"] = df["avg_monthly_rate"] / 30.5
    
    df["days_on_hand"] = (
        df["On_Hand"] / df["avg_daily_rate"].replace(0, np.nan)
    ).fillna(9999)  # 9999 = no sales / effectively infinite
     
     
    def urgent(d):
         if d <= critical_days:
             return "Critical"
         elif d <= warning_days:
             return "Warning"
         return "Okay"
    df["reorder_flag"] = df["days_on_hand"].apply(urgent)  
    
    df["reorder_quantity"] = np.ceil(df["avg_daily_rate"] * reorder_point_days).fillna(0)
    return df


def build_forecast(df : pd.DataFrame) -> pd.DataFrame:
    available = [c for c in month_cols if c in df.columns]
    n_avail = len(available)
    
    records = []
    le = LabelEncoder()
    le.fit(df["SKU"])
    
    for _, row in df.iterrows():
        vals = row[available].values.astype(float)
        sku_enc = le.transform([row["SKU"]])[0]
        
        for i in range(3, n_avail):
            if np.isnan(vals[i]):
                break
            records.append({
                "lag1":    vals[i-1] if not np.isnan(vals[i-1]) else 0,
                "lag2":    vals[i-2] if not np.isnan(vals[i-2]) else 0,
                "lag3":    vals[i-3] if not np.isnan(vals[i-3]) else 0,
                "month_idx": i,
                "sku_enc": sku_enc,
                "target":  vals[i],
            })
        
    if len(records)<20:
        print("[forecast] not enough historical data to train, skips the mL part")
        return df
        
    train = pd.DataFrame(records).dropna(subset=["target"])
    x = train[["lag1", "lag2", "lag3", "month_idx", "sku_enc"]]
    y = train["target"]
        
    model = RandomForestRegressor(n_estimators = 200, random_state = 42, n_jobs = -1)
    model.fit(x, y)
        
    mae = mean_absolute_error(y, model.predict(x))
    print(f"[forecast] Training MAE: {mae:.1f} units/month (in-sample)")
        
    forecast_cols = []
    for _, row in df.iterrows():
        vals = list(row[available].values.astype(float))
        sku_enc = le.transform([row["SKU"]])[0]

        for i in range(n_avail):
            if np.isnan(vals[i]):
                lag1 = vals[i-1] if i >= 1 and not np.isnan(vals[i-1]) else 0
                lag2 = vals[i-2] if i >= 2 and not np.isnan(vals[i-2]) else 0
                lag3 = vals[i-3] if i >= 3 and not np.isnan(vals[i-3]) else 0
                pred = model.predict([[lag1, lag2, lag3, i, sku_enc]])[0]
                vals[i] = max(0, round(pred))

        forecast_cols.append(vals)
        
    forecast_df = pd.DataFrame(
        forecast_cols, 
        columns = [f"fc_{c}" for c in available],
        index = df.index
        )
    df = pd.concat([df, forecast_df], axis= 1)
    
    actual_cols = available
    fc_cols = [f"fc_{c}" for c in available]
        
    def full_year(row):
        total = 0
        for a, f in zip(actual_cols, fc_cols):
            val = row[a]
            total += val if not np.isnan(val) else row[f]
        return total
        
    df["forecast_annual"] = df.apply(full_year, axis = 1)
    df["forecast_avg_monthly"] = df["forecast_annual"] / 12 # maybe // instead of / 
    return df
    
    
urgent_colors = {
    "Critical": "#E05C5C", 
    "Warning":  "#FFD343", 
    "Okay":       "#4CAF82"
}

def plot_top_velocity(df: pd.DataFrame, ax: plt.Axes, top_n: int = 20):
    top = df.nlargest(top_n, "avg_monthly_rate").copy()
    top["SKU_short"] = top["SKU"].str[:22]
    colors = [urgent_colors[f] for f in top["reorder_flag"]]

    ax.barh(top["SKU_short"], top["avg_monthly_rate"], color=colors, edgecolor="white", linewidth=0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Avg Units / Month", fontsize=9)
    ax.set_title(f"Top {top_n} SKUs by Velocity", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)

    patches = [mpatches.Patch(color=v, label=k.capitalize()) for k, v in urgent_colors.items()]
    ax.legend(handles=patches, fontsize=7, loc="lower right")
    
def plot_reorder_heatmap(df: pd.DataFrame, ax: plt.Axes, top_n: int = 40):
    urgent = df[df["days_on_hand"] < 9000].nsmallest(top_n, "days_on_hand").copy()
    if urgent.empty:
        ax.text(0.5, 0.5, "No reorder signals found", ha="center", va="center")
        ax.set_title("Reorder Urgency", fontsize=11, fontweight="bold")
        return
    urgent["SKU_short"] = urgent["SKU"].str[:22]
 
    # Create a single-column heat matrix
    heat_data = urgent[["days_on_hand"]].set_index(urgent["SKU_short"])
 
    sns.heatmap(
        heat_data,
        ax=ax,
        cmap="RdYlGn",
        annot=True,
        fmt=".0f",
        linewidths=0.3,
        cbar_kws={"label": "Days of Stock"},
        annot_kws={"size": 7},
    )
    ax.set_title(f"Days of Stock Remaining — Top {top_n} Most Urgent", fontsize=11, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=7)

    for days, label, color in [
        (critical_days, f"Critical ≤{critical_days}d", "#e74c3c"),
        (warning_days,  f"Warning ≤{warning_days}d",   "#f39c12"),
    ]:
        idx = (urgent["days_on_hand"] > days).idxmax()
        pos = urgent.index.get_loc(idx) if idx in urgent.index else None
        if pos:
            ax.axhline(pos, color=color, linewidth=1.5, linestyle="--", label=label)

    ax.legend(fontsize=7, loc="lower right") 

def plot_forecast_lines(df: pd.DataFrame, ax: plt.Axes, top_n: int = 5):
    fc_cols = [c for c in df.columns if c.startswith("fc_")]
    if not fc_cols:
        ax.text(0.5, 0.5, "Forecast not available", ha="center", va="center")
        ax.set_title("Forecast vs Actuals", fontsize=11, fontweight="bold")
        return
 
    month_labels = [c.replace("fc_", "") for c in fc_cols]
    top = df.nlargest(top_n, "avg_monthly_rate")
 
    cmap = plt.cm.get_cmap("tab10")
    for i, (_, row) in enumerate(top.iterrows()):
        color = cmap(i)
        actuals   = [row[m] if m in df.columns else np.nan for m in month_labels]
        forecasts = [row[f"fc_{m}"] for m in month_labels]
        x = range(len(month_labels))

        ax.plot(x, actuals,   color=color, linewidth=1.8, label=row["SKU"][:18])
        ax.plot(x, forecasts, color=color, linewidth=1.2, linestyle="--", alpha=0.6)

    ax.set_xticks(range(len(month_labels)))
    ax.set_xticklabels(month_labels, rotation=45, fontsize=8)
    ax.set_ylabel("Units Sold / Forecast", fontsize=9)
    ax.set_title(f"Top {top_n} SKUs — Actuals (solid) vs Forecast (dashed)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", ncol=2)
     
    n_actual = df[[c for c in month_cols if c in df.columns]].notna().iloc[0].sum()
    if n_actual < len(month_labels):
        ax.axvspan(n_actual - 0.5, len(month_labels) - 0.5,
                alpha=0.07, color="blue", label="Forecast region")
        

def print_reorder_report(df: pd.DataFrame):
    print("\n" + "="*70)
    print("  REORDER REPORT")
    print(f"  Lead time: {lead_time_days}d  |  Safety buffer: {safety_days}d  "
          f"|  Reorder point: {reorder_point_days}d")
    print("="*70)

    for flag in ["Critical", "Warning"]:
        subset = df[df["reorder_flag"] == flag].sort_values("days_on_hand")
        label  = "🔴 ORDER NOW" if flag == "Critical" else "🟡 ORDER SOON"
        print(f"\n{label}  ({len(subset)} SKUs)\n")
        if subset.empty:
            print("  None")
            continue
        print(f"  {'SKU':<35} {'On Hand':>8} {'Days Left':>10} {'Reorder quantity':>12}")
        print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*12}")
        for _, row in subset.iterrows():
            doh = f"{row['days_on_hand']:.0f}" if row["days_on_hand"] < 9000 else "∞"
            print(f"  {row['SKU']:<35} {int(row['On_Hand']):>8} {doh:>10} {int(row['reorder_quantity']):>12}")

    print("\n" + "="*70 + "\n")
    
def main():
    print("Loading data...")
    df = load_data(csv_path)
    print(f"  {len(df)} SKUs loaded")

    print("Engineering features...")
    df = engineering_features(df)

    print("Building forecast model...")
    df = build_forecast(df)

    print_reorder_report(df)

    print("Rendering dashboard...")
    fig, axes = plt.subplots(1, 3, figsize=(22, 10))
    fig.suptitle("Sales Forecast & Inventory Dashboard", fontsize=15, fontweight="bold", y=1.01)
    fig.patch.set_facecolor("#f8f9fa")
    for ax in axes:
        ax.set_facecolor("#ffffff")

    plot_top_velocity(df, axes[0])
    plot_reorder_heatmap(df, axes[1])
    plot_forecast_lines(df, axes[2])

    plt.tight_layout()
    out_path = "sales_dashboard.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Dashboard saved → {out_path}")

    reorder_df = df[df["reorder_flag"].isin(["Critical","Warning"])][
        ["SKU","On_Hand","avg_daily_rate","days_on_hand","reorder_quantity","reorder_flag"]
    ].sort_values(["reorder_flag","days_on_hand"])
    reorder_df.to_csv("reorder_signals.csv", index=False)
    print("Reorder signals saved → reorder_signals.csv")

if __name__ == "__main__":
    main()