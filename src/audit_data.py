import pandas as pd

files = ["customers.csv", "products.csv", "orders.csv"]

for file in files:
    df = pd.read_csv("data/raw/" + file)

    print("\nFILE:", file)
    print("Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nMissing Percentage:")
    print(df.isnull().mean() * 100)