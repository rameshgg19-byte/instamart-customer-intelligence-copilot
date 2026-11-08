import pandas as pd

orders = pd.read_csv("data/clean/orders.csv")

completed = orders[orders["status"] == "completed"]

amount = completed["order_amount"]

print("Mean:", amount.mean())
print("Median:", amount.median())
print("Skewness:", amount.skew())

q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

print("Q1:", q1)
print("Q3:", q3)
print("Lower Fence:", lower)
print("Upper Fence:", upper)

outliers = completed[
    (amount < lower) | (amount > upper)
]

print("Outliers:", len(outliers))

correlation = completed[
    ["order_amount", "discount_pct", "quantity"]
].corr()

print("\nCorrelation:")
print(correlation)