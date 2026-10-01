import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from common import load_data, set_style, save_fig, save_metrics, CARRIERS, CITY_ORDER, DAY_NAMES

set_style()
df = load_data()
m = {}                                   # numbers quoted in the report

# --- FIG: distributions ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
sns.histplot(df["lead_days"], bins=40, kde=True, color="#2E75B6", ax=ax[0])
ax[0].axvline(df["lead_days"].median(), color="red", ls="--", label="median")
ax[0].set(title="Order-to-delivery lead time", xlabel="days"); ax[0].legend()
sns.histplot(df["freight_cost"], bins=40, kde=True, color="#2E7D32", ax=ax[1])
ax[1].set(title="Freight cost per shipment", xlabel="INR")
sns.histplot(np.log10(df["weight_kg"]), bins=40, kde=True, color="#C55A11", ax=ax[2])
ax[2].set(title="Shipment weight (log10 scale)", xlabel="log10(kg)")
save_fig("fig01_distributions")
# --- END FIG ---

# --- FIG: carrier_boxplots ---------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
sns.boxplot(data=df, x="carrier", y="lead_days", order=CARRIERS, hue="carrier",
            palette="Blues", legend=False, ax=ax[0])
ax[0].set(title="Lead time by carrier", xlabel="", ylabel="days")
sns.boxplot(data=df, x="carrier", y="cost_per_kg", order=CARRIERS, hue="carrier",
            palette="Greens", legend=False, showfliers=False, ax=ax[1])
ax[1].set(title="Cost per kg by carrier (outliers hidden)", xlabel="", ylabel="INR per kg")
save_fig("fig02_carrier_boxplots")
# --- END FIG ---

# --- FIG: correlation_heatmap -----------------------------------------------------------------
cols = ["distance_km", "weight_kg", "order_value", "freight_cost", "processing_hours",
        "transit_days", "delay_days", "customer_rating"]
corr = df[cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))              # show each pair once
plt.figure(figsize=(7.5, 5.8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8})
plt.title("Correlation matrix (Pearson)")
save_fig("fig03_correlation")
# --- END FIG ---

# --- FIG: monthly_trend -------------------------------------------------------------------------
monthly = df.groupby("month").agg(shipments=("shipment_id", "count"),
                                  on_time=("is_late", lambda s: 100 * (1 - s.mean())))
fig, ax1 = plt.subplots(figsize=(8.5, 3.8))
ax1.bar(monthly.index, monthly["shipments"], color="#9DC3E6", label="Shipments")
ax1.set(xlabel="Month (2024)", ylabel="Shipments", xticks=range(1, 13))
ax2 = ax1.twinx()
ax2.plot(monthly.index, monthly["on_time"], color="#C00000", marker="o", label="On-time %")
ax2.set(ylabel="On-time delivery rate (%)", ylim=(60, 100)); ax2.grid(False)
ax1.set_title("Monthly volume versus on-time performance")
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.88))
save_fig("fig04_monthly_trend")
# --- END FIG ---

# --- FIG: lateness_heatmap ------------------------------------------------------------------------
df["hour_band"] = pd.cut(df["order_hour"], [7, 10, 13, 16, 19, 22],
                         labels=["8-10", "11-13", "14-16", "17-19", "20-21"])
late_grid = (df.pivot_table(index="order_dow", columns="hour_band", values="is_late",
                            aggfunc="mean", observed=True) * 100)
late_grid.index = DAY_NAMES
plt.figure(figsize=(7, 4.2))
sns.heatmap(late_grid, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "% late"})
plt.title("Late deliveries (%) by order weekday and hour")
plt.xlabel("Order hour"); plt.ylabel("")
save_fig("fig05_lateness_heatmap")
# --- END FIG ---

# --- FIG: warehouse_bottleneck ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8))
sns.boxplot(data=df, x="order_hour", y="processing_hours", color="#9DC3E6",
            showfliers=False, ax=ax[0])
ax[0].axvline(7.5, color="red", ls="--")                   # between 15:00 and 16:00
ax[0].text(7.4, 31, "4 pm cut-off", color="red", ha="right")
ax[0].set(title="Warehouse processing time by order hour", xlabel="Order hour", ylabel="hours")
# Congestion effect: weekdays and pre-cut-off orders only, so the cut-off and the
# Sunday no-dispatch rule do not hide the volume effect
base = df[(df["order_dow"] < 5) & (df["after_cutoff"] == 0)]
day = base.groupby("order_date").agg(orders=("daily_orders", "first"),
                                     proc=("processing_hours", "median")).reset_index()
sns.regplot(data=day, x="orders", y="proc", scatter_kws={"alpha": .35, "s": 14},
            line_kws={"color": "red"}, ax=ax[1])
ax[1].set(title="Daily order volume vs median processing time\n(Mon-Fri, before cut-off)",
          xlabel="Orders per day", ylabel="median hours")
save_fig("fig06_warehouse_bottleneck")
# --- END FIG ---
m["proc_before_cutoff"] = round(float(df.loc[df.after_cutoff == 0, "processing_hours"].median()), 1)
m["proc_after_cutoff"] = round(float(df.loc[df.after_cutoff == 1, "processing_hours"].median()), 1)
m["corr_daily_proc"] = round(float(day["orders"].corr(day["proc"])), 2)
m["proc_slope_per_10_orders"] = round(float(np.polyfit(day["orders"], day["proc"], 1)[0] * 10), 2)

# --- FIG: cost_drivers ------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8))
sns.boxplot(data=df, x="destination_city", y="freight_cost", order=CITY_ORDER,
            hue="destination_city", palette="viridis", legend=False, ax=ax[0])
ax[0].set(title="Freight cost by destination (ordered by distance)", xlabel="", ylabel="INR")
ax[0].tick_params(axis="x", rotation=20)
sns.scatterplot(data=df.sample(3000, random_state=1), x="weight_kg", y="cost_per_kg",
                hue="carrier", alpha=.5, s=14, ax=ax[1])
ax[1].set(xscale="log", yscale="log", title="Cost per kg falls as weight rises",
          xlabel="Weight (kg, log scale)", ylabel="INR per kg (log scale)")
save_fig("fig07_cost_drivers")
# --- END FIG ---

bands = pd.cut(df["weight_kg"], [0, 10, 30, 100, 500], labels=["<10 kg", "10-30 kg", "30-100 kg", ">100 kg"])
m["cpk_by_weight_band"] = df.groupby(bands, observed=True)["cost_per_kg"].median().round(2).to_dict()
m["shipments_by_weight_band"] = bands.value_counts().sort_index().to_dict()

# --- FIG: carrier_tradeoff ----------------------------------------------------------------------------------------
car = df.groupby("carrier").agg(cost_per_kg=("cost_per_kg", "median"), n=("shipment_id", "count"),
                                on_time=("is_late", lambda s: 100 * (1 - s.mean()))).reset_index()
plt.figure(figsize=(6.8, 4.4))
plt.scatter(car["cost_per_kg"], car["on_time"], s=car["n"] / 6, alpha=.6, color="#2E75B6")
for _, r in car.iterrows():
    plt.annotate(r["carrier"], (r["cost_per_kg"], r["on_time"]),
                 ha="center", va="center", fontsize=9)
plt.xlabel("Median cost per kg (INR)"); plt.ylabel("On-time delivery rate (%)")
plt.title("Carrier trade-off: cost vs reliability (bubble = volume)")
plt.margins(.25)
save_fig("fig08_carrier_tradeoff")
# --- END FIG ---

# --- FIG: carrier_city_heatmap ------------------------------------------------------------------------------------------------
pocket = df.pivot_table(index="carrier", columns="destination_city", values="is_late",
                        aggfunc="mean").loc[CARRIERS, CITY_ORDER] * 100
plt.figure(figsize=(8, 3.6))
sns.heatmap(pocket, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "% late"})
plt.title("Late deliveries (%) by carrier and destination")
plt.xlabel(""); plt.ylabel("")
save_fig("fig09_carrier_city_heatmap")
# --- END FIG ---
m["worst_pocket"] = pocket.stack().idxmax()
m["worst_pocket_pct"] = round(float(pocket.stack().max()), 1)

# --- FIG: pareto_delay_reasons ----------------------------------------------------------------------------------------------------------
late = df[df["is_late"] == 1]["delay_reason"].value_counts()
share = late / late.sum() * 100
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(late.index, late.values, color="#2E75B6")
ax.set(ylabel="Late shipments"); ax.tick_params(axis="x", rotation=15)
ax2 = ax.twinx()
ax2.plot(late.index, share.cumsum(), color="#C00000", marker="o")
ax2.set(ylabel="Cumulative %", ylim=(0, 105)); ax2.grid(False)
ax.set_title("Pareto chart of delay reasons")
save_fig("fig10_pareto_delay_reasons")
# --- END FIG ---
m["delay_reason_share"] = share.round(1).to_dict()

# --- FIG: rating_vs_delay -------------------------------------------------------------------------------------------------------------------
df["delay_band"] = pd.cut(df["delay_days"], [-99, 0, 1, 2, 99],
                          labels=["On time", "1 day late", "2 days late", "3+ days late"])
plt.figure(figsize=(7, 3.8))
sns.barplot(data=df, x="delay_band", y="customer_rating", color="#2E75B6", errorbar=("ci", 95))
plt.ylim(1, 5); plt.xlabel(""); plt.ylabel("Mean customer rating")
plt.title("Customer rating falls with each day of delay")
save_fig("fig11_rating_vs_delay")
# --- END FIG ---
m["rating_by_delay"] = df.groupby("delay_band", observed=True)["customer_rating"].mean().round(2).to_dict()
m["late_by_dow"] = (df.groupby("order_dow")["is_late"].mean() * 100).round(1).to_dict()
m["late_by_month"] = (df.groupby("month")["is_late"].mean() * 100).round(1).to_dict()
m["monthly_shipments"] = monthly["shipments"].to_dict()
m["late_grid"] = late_grid.round(1).to_dict("index")
m["share_orders_after_cutoff"] = round(float(df["after_cutoff"].mean()) * 100, 1)
m["late_share_from_after_cutoff"] = round(float(df.loc[df.after_cutoff == 1, "is_late"].sum()
                                                / df["is_late"].sum()) * 100, 1)
m["orders_by_dow"] = (df["order_dow"].value_counts().sort_index() / len(df) * 100).round(1).to_dict()
m["max_daily_orders"] = int(df["daily_orders"].max())
m["mean_daily_orders"] = round(float(df.groupby("order_date").size().mean()), 1)
m["late_by_cutoff"] = (df.groupby("after_cutoff")["is_late"].mean() * 100).round(1).to_dict()
save_metrics(viz=m)
print("Figures saved:", sorted(p.name for p in __import__("pathlib").Path("figures").glob("*.png")))
