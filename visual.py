import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
import pandas as pd

palette = {
    "critical": "#E05C5C", # this would show up as Red
    "warning":  "#FFD343", # this would show up as Yellow
    "ok":       "#4CAF82" # this would show up as Green
}

status_order = ["critical", "warning", "ok"]
month_cols = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sept", "oct", "nov", "dec"]


def apply_style():
    """Apply a clean, consistent style to all charts."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "figure.facecolor": "#F8F9FA",
        "axes.facecolor":   "#FFFFFF",
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "font.family":      "Times New Roman",
        "axes.titlesize":   15,
        "axes.labelsize":   12,
    })


def plot_velocity_distribution(df: pd.DataFrame) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 12))
    velocities = df["avg_per_mo"].dropna()
    ax.hist(velocities, bins=30, color="#4A90D9", edgecolor="white", linewidth=0.6)

    median_value = velocities.median()
    ax.axvline(median_value, color="#E05C5C", linestyle="--", linewidth=1.5,
               label=f"Median: {median_value:.1f} units/mo")

    ax.set_title("Sales Velocity Distribution (All SKUs)")
    ax.set_xlabel("Avg Units Sold per Month")
    ax.set_ylabel("Number of SKUs")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_stock_status(df: pd.DataFrame, top_n: int = 40) -> plt.Figure:
    apply_style()
    order_map = {"critical": 0, "warning": 1, "ok": 2}
    plot_df = (
        df.assign(_priority=df["reorder_flag"].map(order_map))
          .sort_values(["_priority", "days_on_hand"])
          .head(top_n)
    )
    colors = plot_df["reorder_flag"].map(palette).tolist()
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    ax.barh(plot_df["sku"], plot_df["days_on_hand"], color=colors)
    legend_elements = [Patch(facecolor=palette[s], label=s.capitalize())
                       for s in status_order]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.set_title(f"Days of Stock Remaining — Top {top_n} Most Urgent SKUs")
    ax.set_xlabel("Days of Inventory On Hand")
    ax.set_ylabel("SKU")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
    fig.tight_layout()
    return fig


def plot_seasonal_heatmap(df: pd.DataFrame, top_n: int = 30) -> plt.Figure:
    apply_style()
    available_months = [m for m in month_cols if m in df.columns]

    top_skus = df.nlargest(top_n, "avg_per_mo")
    heat_data = (
        top_skus.set_index("sku")[available_months]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.32)))
    sns.heatmap(
        heat_data,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="#E0E0E0",
        annot=top_n <= 20,
        fmt=".0f",
        cbar_kws={"label": "Units Sold"},
    )
    ax.set_title(f"Monthly Sales Heatmap — Top {top_n} SKUs by Avg Velocity")
    ax.set_xlabel("Month")
    ax.set_ylabel("SKU")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def plot_reorder_timeline(df: pd.DataFrame) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 7))

    for status in status_order:
        subset = df[df["reorder_flag"] == status]
        ax.scatter(
            subset["days_on_hand"],
            subset["days_to_reorder"],
            c=palette[status],
            label=status.capitalize(),
            alpha=0.75,
            s=60,
            edgecolors="white",
            linewidths=0.5,
        )

    max_val = df[["days_on_hand", "days_to_reorder"]].max().max()
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.4,
            label="Stockout = Reorder Point")
    ax.set_title("Reorder Timeline: Days to Stockout vs. Reorder Trigger")
    ax.set_xlabel("Days of Stock Remaining")
    ax.set_ylabel("Days Until Reorder Point")
    ax.legend()

    danger_days_threshold = 30
    danger = df[(df["reorder_flag"] == "critical") & (df["days_on_hand"] < danger_days_threshold)]
    for row in danger.itertuples(index=False):
        ax.annotate(
            row.sku,
            (row.days_on_hand, row.days_to_reorder),
            fontsize=7, alpha=0.8,
            xytext=(4, 4), textcoords="offset points",
        )

    fig.tight_layout()
    return fig


def plot_monthly_trend(df: pd.DataFrame, sku: str) -> plt.Figure:
    apply_style()

    row = df[df["sku"] == sku]
    if row.empty:
        raise ValueError(f"SKU '{sku}' not found in DataFrame.")

    available_months = [m for m in month_cols if m in df.columns]
    values = row[available_months].iloc[0].apply(pd.to_numeric, errors="coerce").values
    x = np.arange(len(available_months))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, values, marker="o", linewidth=2, color="#4A90D9",
            markersize=7, label="Actual Sales")
    ax.fill_between(x, values, alpha=0.12, color="#4A90D9")

    valid_mask = ~np.isnan(values)
    if valid_mask.sum() >= 2:
        coeffs = np.polyfit(x[valid_mask], values[valid_mask], 1)
        trend = np.polyval(coeffs, x)
        direction = "↑" if coeffs[0] > 0 else "↓"
        ax.plot(x, trend, "--", color="#E05C5C", linewidth=1.5,
                label=f"Trend {direction} {abs(coeffs[0]):.1f} units/mo")

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in available_months])
    ax.set_title(f"Monthly Sales Trend — {sku}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Units Sold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_reorder_summary_table(df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    apply_style()

    cols = ["sku", "on_hand", "avg_per_mo", "days_on_hand", "days_to_reorder", "reorder_flag"]
    display_cols = ["SKU", "On Hand", "Avg/Mo", "Days Left", "Order In (days)", "Status"]

    order_map = {"critical": 0, "warning": 1, "ok": 2}
    table_df = (
        df.assign(_priority=df["reorder_flag"].map(order_map))
          .sort_values(["_priority", "days_on_hand"])
          .head(top_n)[cols]
          .copy()
    )
    table_df.columns = display_cols

    for col in ["On Hand", "Avg/Mo", "Days Left", "Order In (days)"]:
        table_df[col] = table_df[col].round(1)

    fig, ax = plt.subplots(figsize=(12, max(4, top_n * 0.38)))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    for i, (_, row) in enumerate(table_df.iterrows(), start=1):
        status = row["Status"]
        bg = {"critical": "#FFE5E5", "warning": "#FFF5E0", "ok": "#E8F7EF"}.get(status, "white")
        for j in range(len(display_cols)):
            tbl[i, j].set_facecolor(bg)

    for j in range(len(display_cols)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(f"Reorder Priority — Top {top_n} SKUs", pad=16, fontsize=13)
    fig.tight_layout()
    return fig