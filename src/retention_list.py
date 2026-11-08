import pandas as pd
import joblib
from pathlib import Path

orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])

cutoff = pd.Timestamp("2025-09-01")

before = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] < cutoff)
]

features = before.groupby("customer_id").agg(
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum"),
    avg_order_value=("order_amount", "mean"),
    avg_discount=("discount_pct", "mean")
).reset_index()

features["last_order"] = (
    before.groupby("customer_id")["order_ts"].max().values
)

features["recency"] = (
    cutoff - features["last_order"]
).dt.days

features = features.drop(columns=["last_order"])

features = features.merge(
    customers,
    on="customer_id",
    how="left"
)

customer_ids = features["customer_id"]

X = features.drop(columns=["customer_id"])

model = joblib.load("outputs/lapse_model.joblib")

probabilities = model.predict_proba(X)[:, 1]

retention = pd.DataFrame({
    "customer_id": customer_ids,
    "lapse_probability": probabilities,
    "predicted_lapsed": (probabilities >= 0.5).astype(int)
})

retention = retention.sort_values(
    "lapse_probability",
    ascending=False
)

Path("outputs").mkdir(exist_ok=True)

retention.to_csv(
    "outputs/retention_list.csv",
    index=False
)

print(retention.head(10))
print("\nTotal Customers:", len(retention))
print("High Risk Customers:", retention["predicted_lapsed"].sum())
print("\nSaved: outputs/retention_list.csv")
print("RETENTION LIST COMPLETED")