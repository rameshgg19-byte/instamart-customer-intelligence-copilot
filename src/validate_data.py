import pandas as pd
import sqlite3

customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

checks = [
    len(customers) == 1200,
    len(products) == 40,
    len(orders) == 12309,
    (orders["status"] == "completed").sum() == 11110,
    (orders["status"] == "cancelled").sum() == 1199,
    orders["order_amount"].isna().sum() == 0,
    orders.loc[orders["status"] == "completed", "delivery_minutes"].isna().sum() == 0,
    orders.loc[orders["status"] == "cancelled", "delivery_minutes"].isna().all(),
    customers["age"].isna().sum() == 0,
    customers["city"].nunique() == 6
]

for i, result in enumerate(checks, 1):
    print(f"Check {i}: {result}")

if all(checks):
    print("ALL CHECKPOINTS PASSED")
else:
    print("CHECK FAILED")