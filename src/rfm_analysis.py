import pandas as pd

orders = pd.read_csv("data/clean/orders.csv")
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

completed = orders[orders["status"] == "completed"]

snapshot_date = pd.Timestamp("2025-10-01")

rfm = completed.groupby("customer_id").agg(
    recency=("order_ts", lambda x: (snapshot_date - x.max()).days),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum")
).reset_index()

print("Total Customers:", len(rfm))
print("\nRFM Data:")
print(rfm.head(10))

print("\nRFM Summary:")
print(rfm[["recency", "frequency", "monetary"]].describe())

rfm.to_csv("data/clean/rfm.csv", index=False)

print("\nRFM ANALYSIS COMPLETED")