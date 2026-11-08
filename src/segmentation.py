import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

rfm = pd.read_csv("data/clean/rfm.csv")

features = rfm[["recency", "frequency", "monetary"]]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

model = KMeans(n_clusters=4, n_init=10, random_state=42)
rfm["segment_id"] = model.fit_predict(scaled_data)

summary = rfm.groupby("segment_id").agg(
    customers=("customer_id", "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean")
).round(2)

print("CUSTOMER SEGMENT SUMMARY")
print(summary)

# Assign segment names based on customer behavior
names = {}

for segment, row in summary.iterrows():
    if row["avg_monetary"] > 5000:
        names[segment] = "High Value"
    elif row["avg_frequency"] > 15:
        names[segment] = "Loyal Customers"
    elif row["avg_recency"] > 60:
        names[segment] = "At Risk"
    else:
        names[segment] = "Regular Customers"

rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_name"] == "High Value"

rfm.to_csv("outputs/customer_segments.csv", index=False)

pca = PCA(n_components=2)
pca.fit(scaled_data)

print("\nPCA EXPLAINED VARIANCE")
print(pca.explained_variance_ratio_)
print("Total:", pca.explained_variance_ratio_.sum())

print("\nSaved: outputs/customer_segments.csv")
print("CUSTOMER SEGMENTATION COMPLETED")