import pandas as pd
from common import load_data, save_metrics

df = load_data()
print("Shape:", df.shape)
print(df.dtypes)

# --- BLOCK: kpis -------------------------------------------------------------------------------
kpis = {
    "shipments": len(df),
    "total_freight_cost": df["freight_cost"].sum(),
    "avg_cost_per_kg": df["freight_cost"].sum() / df["weight_kg"].sum(),
    "avg_lead_days": df["lead_days"].mean(),
    "on_time_rate_pct": 100 * (1 - df["is_late"].mean()),
    "avg_rating": df["customer_rating"].mean(),
}
print(pd.Series(kpis).round(2))
# --- END BLOCK ---

# --- BLOCK: descriptive ------------------------------------------------------------------------
num = ["distance_km", "weight_kg", "units", "order_value", "freight_cost",
       "processing_hours", "transit_days", "lead_days", "delay_days"]
desc = df[num].agg(["mean", "median", "std", "min", "max", "skew"]).T
desc["iqr"] = df[num].quantile(.75) - df[num].quantile(.25)
desc["cv_%"] = 100 * desc["std"] / desc["mean"]          # coefficient of variation
print(desc.round(2))
desc.round(2).to_csv("reports/descriptive_stats.csv")
# --- END BLOCK ---

# --- BLOCK: correlation ------------------------------------------------------------------------
cols = ["distance_km", "weight_kg", "order_value", "freight_cost", "processing_hours",
        "transit_days", "delay_days", "customer_rating"]
pearson, spearman = df[cols].corr(), df[cols].corr(method="spearman")
print(pearson["freight_cost"].round(2))
print(pearson["customer_rating"].round(2))
# --- END BLOCK ---

# --- BLOCK: groups -----------------------------------------------------------------------------
by_carrier = df.groupby("carrier").agg(
    shipments=("shipment_id", "count"), cost_per_kg=("cost_per_kg", "median"),
    avg_lead=("lead_days", "mean"), late_pct=("is_late", lambda s: 100 * s.mean()))
by_city = df.groupby("destination_city").agg(
    shipments=("shipment_id", "count"), avg_cost=("freight_cost", "mean"),
    avg_lead=("lead_days", "mean"), late_pct=("is_late", lambda s: 100 * s.mean()))
print(by_carrier.round(2), by_city.round(2), sep="\n\n")
# --- END BLOCK ---

save_metrics(kpis=kpis, descriptive=desc.round(2).to_dict("index"),
             corr_freight=pearson["freight_cost"].round(2).to_dict(),
             corr_rating=pearson["customer_rating"].round(2).to_dict(),
             spearman_freight_weight=round(float(spearman.loc["freight_cost", "weight_kg"]), 2),
             pearson_freight_weight=round(float(pearson.loc["freight_cost", "weight_kg"]), 2),
             by_carrier=by_carrier.round(2).to_dict("index"),
             by_city=by_city.round(2).to_dict("index"),
             category_counts=df["product_category"].value_counts().to_dict(),
             date_min=str(df["order_date"].min().date()), date_max=str(df["order_date"].max().date()))
