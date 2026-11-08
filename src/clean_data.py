import pandas as pd
import sqlite3
import os

os.makedirs("data/clean", exist_ok=True)

customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")

customers["city"] = customers["city"].str.strip().str.title()

customers["gender"] = customers["gender"].str.strip().str.lower()
customers["gender"] = customers["gender"].replace({
    "m": "male",
    "f": "female",
    "man": "male",
    "woman": "female"
})
customers["gender"] = customers["gender"].where(
    customers["gender"].isin(["male", "female"]), "unknown"
)

customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[~customers["age"].between(10, 100), "age"] = None
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)

customers["signup_date"] = pd.to_datetime(customers["signup_date"])

orders["order_ts"] = pd.to_datetime(orders["order_ts"])

orders = orders.drop_duplicates()
orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates("order_id", keep="last")

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left"
)

missing = orders["order_amount"].isna()

orders.loc[missing, "order_amount"] = (
    orders.loc[missing, "price_inr"]
    * orders.loc[missing, "quantity"]
    * (1 - orders.loc[missing, "discount_pct"] / 100)
).round(2)

orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = None

completed = orders["status"].str.lower() == "completed"

orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed]
    .groupby("category")["delivery_minutes"]
    .transform(lambda x: x.fillna(x.median()))
)

orders = orders.drop(columns=["price_inr", "category"])

customers.to_csv("data/clean/customers.csv", index=False)
products.to_csv("data/clean/products.csv", index=False)
orders.to_csv("data/clean/orders.csv", index=False)

conn = sqlite3.connect("data/instamart.db")

customers.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders.to_sql("orders", conn, if_exists="replace", index=False)

conn.close()

print("DATA CLEANING COMPLETED")
print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))
print(orders["status"].value_counts())