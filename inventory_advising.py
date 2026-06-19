import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

SAFETY_BUFFER_DAYS: int = 7
DEFAULT_LEAD_TIMES_DAYS: int = 14
reorder_alerts: pd.DataFrame

LEAD_TIMES: dict[str, int] = {
    # "NR100134-FCS-AW Zoey": 21
}

MONTH_COLS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sept","Oct","Nov","Dec"]

CURRENT_YEAR_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]

def load_data(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_data = pd.read_csv(filepath, header=None, dtype=str, encoding="utf-8-sig")
    
    header_rows = raw_data.index[raw_data[0].str.contains("Sku", na = False).tolist()]
    
    frames = []
    for i, start in enumerate(header_rows):
        end = header_rows[i + 1] if i + 1 < len(header_rows) else len(raw_data)
        chunk = raw_data.iloc[start:end].copy()
        chunk.columns = chunk.iloc[0]
        chunk = chunk.iloc[1:].reset_index(drop=True)
        
        chunk = chunk.loc[:, chunk.columns.notna() & (chunk.columns != "")]
        chunk = chunk.dropna(subset=[chunk.columns[0]])
        chunk = chunk[chunk.iloc[:, 0].str.strip() != ""]
        frames.append(chunk)
        
    if len(frames) == 0:
        raise ValueError("Couldn't find any data tables in the CSV file")
        
    def normalise(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns= lambda c: c.strip())
        sku_col = [c for c in df.columns if c.lower().startswith("sku")][0]
        df = df.rename(columns={sku_col: "SKU"})
        df["SKU"] = df["SKU"].str.strip()
        for col in MONTH_COLS + ["TOTAL SALES", "Avg per mo", "On Hand 1/1/25", "Months of supply"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
        
    current_df = normalise(frames[0])
    historical_df = normalise(frames[1]) if len(frames) > 1 else pd.DataFrame()
    return current_df, historical_df

def calc_sell_rate(df: pd.DataFrame, month_cols: list[str] | None = None) -> pd.Series:
    if month_cols is None:
        month_cols = [c for c in MONTH_COLS if c in df.columns]
    present = [c for c in month_cols if c in df.columns]
    if not present:
        raise ValueError(f"None of the expected month columns found. Got: {df.columns.tolist()}")
    monthly_sales = df.set_index("SKU")[present]
    avg_monthly = monthly_sales.mean(axis=1, skipna=True).fillna(0)
    daily_rate = avg_monthly / 30.4
    return daily_rate.rename("daily_sell_rate")

def forecast_sell_rate(df: pd.DataFrame, sku: str, month_cols: list[str] | None = None) -> float:
    if month_cols is None:
        month_cols = [c for c in MONTH_COLS if c in df.columns]

    row = df[df["SKU"] == sku]
    if row.empty:
        return 0.0

    values = row[month_cols].iloc[0].dropna().values.astype(float)
    if len(values) < 3:
        return float(values.mean() / 30.4) if len(values) > 0 else 0.0

    X = np.arange(len(values)).reshape(-1, 1)
    y = values
    model = LinearRegression().fit(X, y)
    next_month = model.predict([[len(values)]])[0]
    return max(next_month, 0.0) / 30.4

def calc_lead_time(sku: str):
    return LEAD_TIMES.get(sku, DEFAULT_LEAD_TIMES_DAYS)

def reorder_point(daily_sell_rate: float, 
                  lead_time_days: int) -> float:
    return daily_sell_rate * lead_time_days

def add_safety_stock(daily_sell_rate: float, 
                     buffer_days: int = SAFETY_BUFFER_DAYS) -> float:
    return daily_sell_rate * buffer_days

def effective_reorder_point(daily_sell_rate: float, 
                             lead_time_days: int, 
                             buffer_days: int = SAFETY_BUFFER_DAYS):
    return daily_sell_rate * (lead_time_days + buffer_days)

def order_by_date(on_hand: float,
                  daily_sell_rate: float,
                  lead_time_days: int,
                  buffer_days: int = SAFETY_BUFFER_DAYS,
                  today: date | None = None) -> date | None:
    if today is None:
        today = date.today()
    if pd.isna(on_hand) or on_hand <= 0 or daily_sell_rate <= 0:
        return None
    
    safety_stock = add_safety_stock(daily_sell_rate, buffer_days)
    usable_stock = on_hand - safety_stock
    if usable_stock <= 0:
        return today
    
    days_until_reorder = usable_stock / daily_sell_rate
    order_date = today + timedelta(days=int(days_until_reorder) - lead_time_days)
    return order_date

def build_reorder_alerts(df: pd.DataFrame,
                          month_cols: list[str] | None = None,
                          use_forecast: bool = True) -> pd.DataFrame:
    if month_cols is None:
        month_cols = [c for c in MONTH_COLS if c in df.columns]
    simple_rate = calc_sell_rate(df, month_cols)
    today = date.today()
    records = []
    
    on_hand_col = "On Hand 1/1/25" if "On Hand 1/1/25" in df.columns else None
    
    for _, row in df.iterrows():
        sku = row["SKU"]

        if use_forecast:
            rate = forecast_sell_rate(df, sku, month_cols)
            if rate == 0 and sku in simple_rate.index:
                rate = simple_rate[sku]
        else:
            rate = simple_rate.get(sku, 0.0)

        on_hand = row[on_hand_col] if on_hand_col else np.nan
        lead_time = calc_lead_time(sku)
        safety = add_safety_stock(rate)
        rop = reorder_point(rate, lead_time)
        
        if rate > 0 and not pd.isna(on_hand):
            days_of_stock = on_hand / rate
        else:
            days_of_stock = np.nan

        order_date = order_by_date(on_hand, rate, lead_time, today=today)

        if pd.isna(on_hand):
            urgency = "UNKNOWN"
        elif on_hand <= rop:
            urgency = "CRITICAL"
        elif not pd.isna(days_of_stock) and days_of_stock <= (lead_time + SAFETY_BUFFER_DAYS + 7):
            urgency = "WARNING"
        else:
            urgency = "OK"

        records.append({
            "SKU": sku,
            "on_hand": on_hand,
            "daily_sell_rate": round(rate, 3),
            "lead_time_days": lead_time,
            "safety_stock_units": round(safety, 1),
            "reorder_point_units": round(rop, 1),
            "days_of_stock": round(days_of_stock, 1) if not pd.isna(days_of_stock) else np.nan,
            "order_by": order_date,
            "urgency": urgency,
        })

    alerts = pd.DataFrame(records)
    order_map = {"CRITICAL": 0, "WARNING": 1, "OK": 2, "UNKNOWN": 3}
    alerts["_sort"] = alerts["urgency"].map(order_map)
    alerts = alerts.sort_values(["_sort", "days_of_stock"]).drop(columns="_sort")
    return alerts.reset_index(drop=True)
        
        
PALETTE = {"CRITICAL": "#d62728", "WARNING": "#ff7f0e", "OK": "#2ca02c", "UNKNOWN": "#aec7e8"}

def apply_style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=15, fontname="Times New Roman", pad=10)
    ax.set_xlabel(xlabel, fontsize=12, fontname="Times New Roman")
    ax.set_ylabel(ylabel, fontsize=12, fontname="Times New Roman")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        
def plot_urgency_breakdown(alerts: pd.DataFrame, save_path: str = "urgency_breakdown.png"):
    counts = alerts["urgency"].value_counts().reindex(
        ["CRITICAL", "WARNING", "OK", "UNKNOWN"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[k] for k in counts.index], edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom",
                fontsize=11, fontname="Times New Roman", fontweight="bold")
    apply_style(ax, "Inventory Urgency Breakdown", "Urgency", "# of SKUs")
    ax.set_ylim(0, counts.max() * 1.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")

def plot_days_of_stock(alerts: pd.DataFrame, top_n: int = 25,
                       save_path: str = "days_of_stock.png"):
    subset = alerts.dropna(subset=["days_of_stock"]).nsmallest(top_n, "days_of_stock")
    colors = [PALETTE[u] for u in subset["urgency"]]
    fig, ax = plt.subplots(figsize=(10, max(5, len(subset) * 0.35)))
    ax.barh(subset["SKU"], subset["days_of_stock"], color=colors, edgecolor="white")
    ax.axvline(SAFETY_BUFFER_DAYS + DEFAULT_LEAD_TIMES_DAYS, color="red",
               linestyle="--", linewidth=1.2, label="Lead time + buffer")
    ax.legend(fontsize=10)
    apply_style(ax, f"Top {top_n} SKUs — Lowest Days of Stock",
                 "Days of Stock Remaining", "SKU")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")
    
def plot_monthly_trend(df: pd.DataFrame, skus: list[str],
                       month_cols: list[str] | None = None,
                       save_path: str = "monthly_trend.png"):
    if month_cols is None:
        month_cols = [c for c in MONTH_COLS if c in df.columns]
    fig, ax = plt.subplots(figsize=(10, 5))
    for sku in skus:
        row = df[df["SKU"] == sku]
        if row.empty:
            continue
        vals = row[month_cols].iloc[0].astype(float)
        ax.plot(month_cols, vals, marker="o", label=sku)
    apply_style(ax, "Monthly Sales Trend", "Month", "Units Sold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")
    
def plot_sell_rate_distribution(alerts: pd.DataFrame,
                                save_path: str = "sell_rate_dist.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(alerts["daily_sell_rate"], bins=30, kde=True,
                 color="#1f77b4", edgecolor="white", ax=ax)
    ax.axvline(alerts["daily_sell_rate"].median(), color="red",
               linestyle="--", linewidth=1.2,
               label=f"Median: {alerts['daily_sell_rate'].median():.2f} units/day")
    ax.legend(fontsize=10)
    apply_style(ax, "Daily Sell Rate Distribution", "Units / Day", "# of SKUs")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_order_calendar(alerts: pd.DataFrame, days_ahead: int = 60,
                        save_path: str = "order_calendar.png"):
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    subset = alerts.dropna(subset=["order_by"])
    subset = subset[subset["order_by"].apply(
        lambda d: isinstance(d, date) and today <= d <= cutoff
    )]
    if subset.empty:
        print("  No upcoming orders in the window — skipping order calendar.")
        return
    fig, ax = plt.subplots(figsize=(10, max(4, len(subset) * 0.3)))
    for _, row in subset.iterrows():
        ax.scatter(row["order_by"], row["SKU"],
                   color=PALETTE.get(row["urgency"], "gray"), s=80, zorder=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45)
    ax.axvline(today, color="black", linestyle=":", linewidth=1, label="Today")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    apply_style(ax, f"Order-By Calendar — Next {days_ahead} Days",
                 "Order By Date", "SKU")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def main():
    CSV_PATH = "Sales_by_Month.csv"

    print("Loading data...")
    current_df, historical_df = load_data(CSV_PATH)
    print(f"  Current-year SKUs : {len(current_df)}")
    print(f"  Historical SKUs   : {len(historical_df)}")

    print("\nBuilding reorder alerts from historical data...")
    alerts = build_reorder_alerts(historical_df, use_forecast=True)

    print("\n── Reorder Alert Summary ──")
    print(alerts["urgency"].value_counts().to_string())

    print("\n── CRITICAL SKUs (order NOW) ──")
    critical = alerts[alerts["urgency"] == "CRITICAL"][
        ["SKU", "on_hand", "daily_sell_rate", "reorder_point_units", "days_of_stock", "order_by"]
    ]
    print(critical.to_string(index=False))

    print("\n── WARNING SKUs (order soon) ──")
    warning = alerts[alerts["urgency"] == "WARNING"][
        ["SKU", "on_hand", "days_of_stock", "order_by"]
    ]
    print(warning.to_string(index=False))

    alerts.to_csv("reorder_alerts.csv", index=False)
    print("\nExported → reorder_alerts.csv")

    print("\nGenerating charts...")
    plot_urgency_breakdown(alerts)
    plot_days_of_stock(alerts)
    plot_sell_rate_distribution(alerts)
    plot_order_calendar(alerts)

    top5 = alerts.nlargest(5, "daily_sell_rate")["SKU"].tolist()
    hist_month_cols = [c for c in MONTH_COLS if c in historical_df.columns]
    plot_monthly_trend(historical_df, top5, hist_month_cols)

    print("\nDone.") 

if __name__ == "__main__":
    main()