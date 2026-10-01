"""Simulates a year of shipments for RegionalMart Distribution (one hub, six destinations).
Relationships between variables are built in on purpose, so the analysis can be checked.
The data is assumed to have already passed the Week 2 cleaning pipeline."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)
N = 12000
CITIES = {"Pune": 150, "Hyderabad": 710, "Bengaluru": 980,
          "Chennai": 1330, "Delhi": 1400, "Kolkata": 1960}
CARRIERS = ["SwiftFreight", "RoadLink", "CargoNova", "QuickHaul"]
CATS = ["Grocery", "Beverages", "Personal Care", "Household", "Electronics"]

# ---- 1. When do orders arrive? (weekday pattern, festive peak in Oct-Nov, daily rhythm) -------
days = pd.date_range("2024-01-01", "2024-12-31")
season = {1: .95, 2: .90, 3: 1.0, 4: 1.0, 5: 1.05, 6: .95,
          7: .90, 8: .95, 9: 1.05, 10: 1.45, 11: 1.35, 12: 1.10}
weekday = {0: 1.15, 1: 1.10, 2: 1.05, 3: 1.05, 4: 1.0, 5: 0.80, 6: 0.35}
p_day = np.array([season[d.month] * weekday[d.dayofweek] for d in days])
p_day /= p_day.sum()
day_idx = rng.choice(len(days), N, p=p_day)
hours = np.arange(8, 22)
p_hour = np.array([2, 4, 7, 9, 9, 7, 6, 6, 7, 8, 8, 6, 4, 2], float)
hour = rng.choice(hours, N, p=p_hour / p_hour.sum())
order_dt = (days[day_idx] + pd.to_timedelta(hour, unit="h")
            + pd.to_timedelta(rng.integers(0, 60, N), unit="m"))
daily_orders = pd.Series(day_idx).map(pd.Series(day_idx).value_counts()).to_numpy()

# ---- 2. Product, size and value ----------------------------------------------------------------
cat = rng.choice(CATS, N, p=[.30, .20, .15, .20, .15])
w_par = {"Grocery": (3.2, .7), "Beverages": (3.5, .6), "Personal Care": (2.4, .6),
         "Household": (2.8, .7), "Electronics": (1.8, .7)}
unit_wt = {"Grocery": .8, "Beverages": 1.2, "Personal Care": .4, "Household": .9, "Electronics": 1.5}
price = {"Grocery": 90, "Beverages": 70, "Personal Care": 150, "Household": 180, "Electronics": 1500}
weight = np.array([rng.lognormal(*w_par[c]) for c in cat]).round(1).clip(0.5, 400)
units = np.maximum(1, np.round(weight / np.array([unit_wt[c] for c in cat])
                               * rng.normal(1, .08, N))).astype(int)
order_value = (units * np.array([price[c] for c in cat]) * rng.lognormal(0, .25, N)).round(2)
vehicle = np.where(weight < 40, "Van", np.where(weight < 120, "Mini truck", "Truck"))

# ---- 3. Warehouse processing (bottleneck: daily congestion + 4 pm cut-off) ----------------------
proc = (rng.lognormal(np.log(7), .35, N)
        + 0.12 * np.maximum(daily_orders - 35, 0)
        + (hour >= 16) * rng.normal(13, 2, N))
dispatch_dt = order_dt + pd.to_timedelta(proc, unit="h")
dispatch_dt = dispatch_dt + pd.to_timedelta((dispatch_dt.dayofweek == 6) * 1, unit="D")  # no Sunday dispatch

# ---- 4. Transit (carrier speed and reliability, monsoon, peak-season capacity) -------------------
dest = rng.choice(list(CITIES), N, p=[.20, .15, .20, .15, .18, .12])
dist = (np.array([CITIES[c] for c in dest]) * rng.normal(1, .06, N)).round(1)
carrier = rng.choice(CARRIERS, N, p=[.30, .30, .22, .18])
speed = {"SwiftFreight": 580, "RoadLink": 490, "CargoNova": 440}
spd = np.where(carrier == "QuickHaul", np.where(dist < 500, 650, 450),
               [speed.get(c, 490) for c in carrier])
noise_sd = {"SwiftFreight": .30, "RoadLink": .45, "CargoNova": .70, "QuickHaul": .50}
sd = np.array([noise_sd[c] for c in carrier]) * np.where((carrier == "QuickHaul") & (dist >= 1000), 1.8, 1)
month = dispatch_dt.month.to_numpy()
monsoon = np.isin(month, [7, 8, 9])
peak = np.isin(month, [10, 11])
south_east = np.isin(dest, ["Kolkata", "Chennai"])
transit = (dist / spd + np.abs(rng.normal(0, 1, N)) * sd
           + monsoon * np.where(south_east, rng.normal(.8, .4, N), rng.normal(.25, .3, N))
           + peak * np.isin(carrier, ["CargoNova", "RoadLink"]) * rng.normal(.5, .3, N)).clip(.4, None)
delivery_dt = dispatch_dt + pd.to_timedelta(transit, unit="D")
promised = order_dt.normalize() + pd.to_timedelta(np.ceil(dist / 450) + 1, unit="D")

# ---- 5. Freight cost (distance, weight with economies of scale, carrier rate, peak surcharge) ---
rate = {"SwiftFreight": 1.12, "RoadLink": .98, "CargoNova": .85, "QuickHaul": 1.05}
cost = (45 + np.array([rate[c] for c in carrier]) * (0.052 * dist + 1.4 * weight ** 0.92)
        ) * np.where(peak, 1.12, 1.0) * rng.lognormal(0, .08, N)

# ---- 6. Customer rating and delay reasons -------------------------------------------------------
delay = (pd.Series(delivery_dt).dt.normalize() - pd.Series(promised)).dt.days.to_numpy()
rating = np.clip(np.round(4.6 + rng.normal(0, .5, N) - .45 * np.maximum(delay, 0)), 1, 5).astype(int)
reasons = ["Warehouse backlog", "Carrier capacity", "Transit / road", "Weather", "Address / customer"]
reason = np.full(N, "None", dtype=object)
for i in np.where(delay > 0)[0]:
    w = np.array([.25, .20, .25, .10, .20])
    if proc[i] > 30: w[0] += .35
    if monsoon[i] and south_east[i]: w[3] += .35
    if peak[i] and carrier[i] in ("CargoNova", "RoadLink"): w[1] += .30
    if dist[i] > 1300: w[2] += .15
    reason[i] = rng.choice(reasons, p=w / w.sum())

df = pd.DataFrame({
    "shipment_id": [f"S{i:06d}" for i in range(N)],
    "order_datetime": order_dt, "dispatch_datetime": dispatch_dt,
    "delivery_datetime": delivery_dt, "promised_date": promised,
    "carrier": carrier, "destination_city": dest, "product_category": cat,
    "vehicle_type": vehicle, "distance_km": dist, "units": units, "weight_kg": weight,
    "order_value": order_value, "freight_cost": cost.round(2),
    "customer_rating": rating, "delay_reason": reason})
df = df.sort_values("order_datetime")
df["shipment_id"] = [f"S{i:06d}" for i in range(N)]
df.to_csv("data/shipments.csv", index=False)
print("Dataset written:", df.shape)
