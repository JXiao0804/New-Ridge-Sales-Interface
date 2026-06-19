# importing the necessary libraries
import pandas as pd
import numpy as np

# the wiggle rooms
safety_days = 7
lead_time_days = 14

def load_and_prepare_data():
    """
    runs the full pipeline and returns a clean data frame
    """
    df = load_csv()
    df = clean_data(df)
    df = engineer_feat(df)
    df = compute_reorder_signals(df)
    return df

def load_csv():
    """
    loads in the CSV file
    """
    df = pd.read_csv("Sales by Month.csv", encoding="utf-8-sig")
    return df

def clean_data(df):
    """
    cleans the file by column names, drops bad rows, and casts all columns to the correct types
    """
    df.columns = (df.columns.str.strip().str.lower().str.replace(" ", "_"))
    df = df.rename(columns={
        "sku_2026":"sku",
        "on_hand_1/1/25": "on_hand",
        })
    df = df.dropna(subset = ["sku"])
    df = df[df["sku"].str.strip() != ""]
    month_cols = ["jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sept", "oct", "nov", "dec"]
    df[month_cols] = df[month_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    df["total_sales"] = pd.to_numeric(df["total_sales"], errors="coerce").fillna(0)
    df["avg_per_mo"] = pd.to_numeric(df["avg_per_mo"], errors="coerce").fillna(0)
    df["on_hand"] = pd.to_numeric(df["on_hand"], errors="coerce").fillna(0)
    df["months_of_supply"] = pd.to_numeric(df["months_of_supply"], errors="coerce").fillna(0)
    
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]
    df = df.reset_index(drop=True)
    return df

def engineer_feat(df):
    """Adds active_months, avg_daily_sales, and days_of_stock columns to the DataFrame."""
    month_cols = ["jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sept", "oct", "nov", "dec"]

    df["active_months"] = (df[month_cols] > 0).sum(axis=1)

    df["total_sales"] = df[month_cols].sum(axis=1)
    df["avg_per_mo"]  = df.apply(
        lambda row: row["total_sales"] / row["active_months"] if row["active_months"] > 0 else 0,
        axis=1
    )

    df["avg_daily_sales"] = df["avg_per_mo"] / 30
    
    df["days_of_stock"] = np.where(
        df["avg_daily_sales"] > 0,
        df["on_hand"] / df["avg_daily_sales"],
        np.inf
    )

    df["months_of_supply"] = np.where(
        df["avg_per_mo"] > 0,
        df["on_hand"] / df["avg_per_mo"],
        np.inf
    )
    return df
    
def compute_reorder_signals(df):
    """
    flags each SKU with whether it needs reordering and how many days 
    until the reorder window given the wiggle room days in case of a mistake
    """
    df["reorder_point_days"] =lead_time_days + safety_days
    df["should_reorder"] = df["days_of_stock"] < df["reorder_point_days"]
    df["days_until_reorder"] = df["days_of_stock"] - df["reorder_point_days"]
    return df

def get_order_alerts(df):
    """
    Returns only the SKUs that need ordering right now, sorted most urgent first.
    """
    alerts = df[df["should_reorder"] == True].copy()
    alerts = alerts.sort_values("days_until_reorder")
    return alerts[["sku", "on_hand", "avg_daily_sales",
                   "days_of_stock", "reorder_point_days", "days_until_reorder"]]

def summary(df):
    """
    returns a per-SKU summary table for use in dashboard cards
    """
    return df[["sku", 
              "total_sales", 
              "avg_per_mo", 
              "on_hand", 
              "months_of_supply", 
              "days_of_stock"]].copy()
    