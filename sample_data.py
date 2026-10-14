"""A small, deterministic sales table for the demo and the tests."""
import pandas as pd

REGIONS = ["North", "South", "East", "West"]
PRODUCTS = ["laptop", "phone", "tablet", "monitor", "dock"]
PRICES = {"laptop": 900, "phone": 600, "tablet": 350, "monitor": 220, "dock": 90}


def make_sales():
    rows = []
    for n in range(60):
        product = PRODUCTS[(n * 3) % 5]
        rows.append({"date": f"2026-0{1 + n % 9}-{1 + n % 27:02d}",
                     "region": REGIONS[n % 4],
                     "product": product,
                     "units": 1 + (n * 7) % 9,
                     "price": PRICES[product]})
    df = pd.DataFrame(rows)
    df["revenue"] = df["units"] * df["price"]
    return df
