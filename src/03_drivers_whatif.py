import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
from common import load_data, set_style, save_fig, save_metrics

set_style()
df = load_data()
m = {}

# --- BLOCK: cost_model -------------------------------------------------------------------------
# What drives freight cost? Multiple linear regression (OLS)
ols = smf.ols("freight_cost ~ distance_km + weight_kg + C(carrier, Treatment('SwiftFreight'))"
              " + peak_season", data=df).fit()
print(ols.summary().tables[1])
print("R-squared:", round(ols.rsquared, 3))
# --- END BLOCK ---
ci = ols.conf_int()
m["ols_r2"] = round(float(ols.rsquared), 3)
m["ols_coef"] = {k: round(float(v), 3) for k, v in ols.params.items()}
m["ols_p_max"] = float(ols.pvalues.max())

# --- BLOCK: late_model ----------------------------------------------------------------------------
# What drives late delivery? Logistic regression using only information known at order time
df["sat_order"] = (df["order_dow"] == 5).astype(int)
df["sun_order"] = (df["order_dow"] == 6).astype(int)
df["orders_per_10"] = df["daily_orders"] / 10
df["distance_100km"] = df["distance_km"] / 100
formula = ("is_late ~ after_cutoff + sat_order + sun_order + orders_per_10 + distance_100km"
           " + peak_season + monsoon + C(carrier, Treatment('SwiftFreight'))")
df = df.sort_values("order_datetime")
cut = int(len(df) * 0.8)                                    # time-ordered split
train, test = df.iloc[:cut], df.iloc[cut:]
logit = smf.logit(formula, data=train).fit(disp=0)
auc = roc_auc_score(test["is_late"], logit.predict(test))
print(f"Test AUC: {auc:.3f}")
odds = pd.DataFrame({"odds_ratio": np.exp(logit.params),
                     "low": np.exp(logit.conf_int()[0]),
                     "high": np.exp(logit.conf_int()[1])}).drop("Intercept")
print(odds.round(2))
# --- END BLOCK ---
m["logit_auc"] = round(float(auc), 3)
m["odds"] = odds.round(2).to_dict("index")

# --- FIG: odds_ratios ----------------------------------------------------------------------------------
labels = {"after_cutoff": "Order after 4 pm", "sat_order": "Saturday order",
          "sun_order": "Sunday order", "orders_per_10": "+10 orders that day",
          "distance_100km": "+100 km distance", "peak_season": "Peak season (Oct-Nov)",
          "monsoon": "Monsoon (Jul-Sep)"}
carrier_label = lambda i: i.split("[T.")[-1].rstrip("]") + " (vs SwiftFreight)"
odds.index = [labels.get(i, carrier_label(i)) for i in odds.index]
odds = odds.sort_values("odds_ratio")
plt.figure(figsize=(7.5, 4.2))
plt.errorbar(odds["odds_ratio"], range(len(odds)),
             xerr=[odds["odds_ratio"] - odds["low"], odds["high"] - odds["odds_ratio"]],
             fmt="o", color="#2E75B6", ecolor="#9DC3E6", capsize=3)
plt.axvline(1, color="red", ls="--"); plt.xscale("log")
plt.yticks(range(len(odds)), odds.index)
plt.xlabel("Odds ratio of a late delivery (log scale, 95% CI)")
plt.title("Drivers of late delivery")
save_fig("fig12_odds_ratios")
# --- END FIG ---

# --- BLOCK: whatif -------------------------------------------------------------------------------------------
def on_time_pct(frame, hours_saved):
    """On-time % if each shipment were delivered `hours_saved` earlier."""
    new_delivery = frame["delivery_datetime"] - pd.to_timedelta(hours_saved, unit="h")
    late = (new_delivery.dt.normalize() - frame["promised_date"]).dt.days > 0
    return 100 * (1 - late.mean())

base = on_time_pct(df, 0)
# A. Same-day dispatch for orders after the 4 pm cut-off
gap = df.loc[df.after_cutoff == 1, "processing_hours"].median() \
      - df.loc[df.after_cutoff == 0, "processing_hours"].median()
save_a = np.where(df["after_cutoff"] == 1, np.minimum(gap, df["processing_hours"] - 2), 0)
# B. Dispatch on Sundays: weekend orders that were pushed to Monday save 24 hours
pushed = df["order_dow"].isin([5, 6]) & (df["dispatch_datetime"].dt.dayofweek == 0)
save_b = np.where(pushed, 24, 0)
res = {"Baseline": base, "A: cut-off fix": on_time_pct(df, save_a),
       "B: Sunday dispatch": on_time_pct(df, save_b),
       "A + B combined": on_time_pct(df, save_a + save_b)}

# C. Move high-late lanes from CargoNova / QuickHaul to SwiftFreight (cost vs service)
lane = (df.groupby(["carrier", "destination_city"])
          .agg(n=("is_late", "size"), late=("is_late", "mean"),
               cost=("freight_cost", "sum"), kg=("weight_kg", "sum")).reset_index())
swift = lane[lane["carrier"] == "SwiftFreight"].set_index("destination_city")
pockets = lane[(lane["carrier"] != "SwiftFreight") & (lane["late"] > 0.25)].copy()
pockets["swift_late"] = pockets["destination_city"].map(swift["late"])
pockets["avoided"] = pockets["n"] * (pockets["late"] - pockets["swift_late"])
pockets["cost_ratio"] = (pockets["destination_city"].map(swift["cost"] / swift["kg"])
                         / (pockets["cost"] / pockets["kg"]))
pockets["extra_cost"] = pockets["cost"] * (pockets["cost_ratio"] - 1)
total_late = df["is_late"].sum()
res["C: lane reallocation"] = base + 100 * pockets["avoided"].sum() / len(df)
print(pockets.round(2))
print({k: round(v, 1) for k, v in res.items()})
# --- END BLOCK ---
m["whatif"] = {k: round(float(v), 1) for k, v in res.items()}
m["gap_hours"] = round(float(gap), 1)
m["pushed_shipments"] = int(pushed.sum())
m["pockets"] = pockets.round(2).to_dict("records")
m["lane_avoided"] = round(float(pockets["avoided"].sum()), 0)
m["lane_extra_cost"] = round(float(pockets["extra_cost"].sum()), 0)
m["lane_cost_per_avoided"] = round(float(pockets["extra_cost"].sum() / pockets["avoided"].sum()), 0)
m["lane_shipments"] = int(pockets["n"].sum())
m["total_cost"] = round(float(df["freight_cost"].sum()), 0)
m["total_late"] = int(total_late)

# --- FIG: whatif ------------------------------------------------------------------------------------------------
plt.figure(figsize=(7.5, 3.8))
names, vals = list(res), list(res.values())
bars = plt.bar(names, vals, color=["#7F7F7F", "#2E75B6", "#2E75B6", "#1F3864", "#2E7D32"])
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v + .3, f"{v:.1f}%", ha="center")
plt.ylim(min(vals) - 5, 100); plt.ylabel("On-time delivery rate (%)")
plt.xticks(rotation=12); plt.title("Estimated on-time rate under improvement scenarios")
save_fig("fig13_whatif")
# --- END FIG ---
save_metrics(models=m)
