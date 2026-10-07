"""Generates a deliberately messy retail sales dataset for the demo (sales_data.csv).

Problems planted on purpose, so the auto-profiling and cleaning steps have something to find:
- Order_Date stored as text (DD-MM-YYYY) instead of a date type
- Missing values in Region, Sales and Customer_Segment
- Inconsistent Region labels ("north", "North ", "NORTH")
- ~40 exact duplicate rows
- Furniture > Tables sold at heavy discounts, which drives negative profit (the "why" question)
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 2000

catalog = {
    "Furniture": {"Chairs": 220, "Tables": 480, "Bookcases": 300, "Furnishings": 60},
    "Technology": {"Phones": 350, "Laptops": 900, "Accessories": 80, "Printers": 260},
    "Office Supplies": {"Paper": 25, "Binders": 30, "Storage": 120, "Art": 15},
}
regions = ["North", "South", "East", "West", "Central"]
region_weights = [0.24, 0.18, 0.22, 0.26, 0.10]
segments = ["Consumer", "Corporate", "Small Business"]

rows = []
dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
for i in range(N):
    cat = rng.choice(list(catalog), p=[0.25, 0.30, 0.45])
    sub = rng.choice(list(catalog[cat]))
    base = catalog[cat][sub]
    qty = int(rng.integers(1, 9))
    # Seasonality: Q4 lifts sales; growth trend across 2024 -> 2025
    d = dates[rng.integers(0, len(dates))]
    season = 1.35 if d.month in (10, 11, 12) else 1.0
    growth = 1.12 if d.year == 2025 else 1.0
    price = base * rng.uniform(0.8, 1.2) * season * growth
    discount = rng.choice([0, 0.1, 0.2], p=[0.6, 0.3, 0.1])
    if sub == "Tables":
        discount = rng.choice([0.3, 0.4, 0.5], p=[0.3, 0.4, 0.3])  # the hidden problem
    sales = round(price * qty * (1 - discount), 2)
    margin = rng.uniform(0.15, 0.35) - discount * 0.9
    profit = round(sales * margin, 2)
    rows.append({
        "Order_ID": f"ORD-{10000 + i}",
        "Order_Date": d.strftime("%d-%m-%Y"),
        "Region": rng.choice(regions, p=region_weights),
        "Customer_Segment": rng.choice(segments, p=[0.5, 0.3, 0.2]),
        "Category": cat,
        "Sub_Category": sub,
        "Quantity": qty,
        "Discount": discount,
        "Sales": sales,
        "Profit": profit,
    })

df = pd.DataFrame(rows)

# Plant data-quality problems
idx = rng.choice(N, 90, replace=False)
df.loc[idx[:60], "Region"] = np.nan
df.loc[idx[60:], "Sales"] = np.nan
df.loc[rng.choice(N, 35, replace=False), "Customer_Segment"] = np.nan
messy = rng.choice(df.index[df["Region"] == "North"], 70, replace=False)
df.loc[messy[:25], "Region"] = "north"
df.loc[messy[25:50], "Region"] = "North "
df.loc[messy[50:], "Region"] = "NORTH"
df = pd.concat([df, df.sample(40, random_state=7)], ignore_index=True)
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

df.to_csv("sales_data.csv", index=False)
print(f"Wrote sales_data.csv with {len(df)} rows")
