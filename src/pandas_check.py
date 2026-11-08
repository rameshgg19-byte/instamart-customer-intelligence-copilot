import pandas as pd

customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

completed = orders[orders["status"] == "completed"]

merged = completed.merge(customers, on="customer_id")
merged = merged.merge(products, on="product_id")

pivot = pd.pivot_table(
    merged,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True
)

print(pivot)

print("\nTotal Revenue:", pivot.loc["All", "All"])