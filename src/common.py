"""Shared helpers: data loading with derived columns, plotting style, metric storage."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SUMMARY = Path("reports/summary.json")
CARRIERS = ["SwiftFreight", "RoadLink", "CargoNova", "QuickHaul"]
CITY_ORDER = ["Pune", "Hyderabad", "Bengaluru", "Chennai", "Delhi", "Kolkata"]  # by distance
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def set_style():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 170,
                         "axes.titleweight": "bold", "axes.titlesize": 11})


def save_fig(name):
    Path("figures").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", bbox_inches="tight", facecolor="white")
    plt.close()


def save_metrics(**kwargs):
    SUMMARY.parent.mkdir(exist_ok=True)
    data = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    data.update(kwargs)
    SUMMARY.write_text(json.dumps(data, indent=2,
                       default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def load_data(path="data/shipments.csv"):
    """Load the shipment file and add the derived analytical columns."""
    df = pd.read_csv(path, parse_dates=["order_datetime", "dispatch_datetime",
                                        "delivery_datetime", "promised_date"])
    # Time-based features
    df["order_date"] = df["order_datetime"].dt.normalize()
    df["order_hour"] = df["order_datetime"].dt.hour
    df["order_dow"] = df["order_datetime"].dt.dayofweek
    df["month"] = df["order_datetime"].dt.month
    # Process durations
    hours = lambda delta: delta.dt.total_seconds() / 3600
    df["processing_hours"] = hours(df["dispatch_datetime"] - df["order_datetime"])
    df["transit_days"] = hours(df["delivery_datetime"] - df["dispatch_datetime"]) / 24
    df["lead_days"] = hours(df["delivery_datetime"] - df["order_datetime"]) / 24
    # Service level: late if delivered after the promised date
    df["delay_days"] = (df["delivery_datetime"].dt.normalize() - df["promised_date"]).dt.days
    df["is_late"] = (df["delay_days"] > 0).astype(int)
    # Cost efficiency
    df["cost_per_kg"] = df["freight_cost"] / df["weight_kg"]
    df["cost_per_km"] = df["freight_cost"] / df["distance_km"]
    # Context flags
    df["peak_season"] = df["month"].isin([10, 11]).astype(int)
    df["monsoon"] = df["month"].isin([7, 8, 9]).astype(int)
    df["after_cutoff"] = (df["order_hour"] >= 16).astype(int)
    df["daily_orders"] = df.groupby("order_date")["shipment_id"].transform("count")
    return df
