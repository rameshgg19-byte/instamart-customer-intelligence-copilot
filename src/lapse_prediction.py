import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score

orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])

cutoff = pd.Timestamp("2025-09-01")

before = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] < cutoff)
]

after = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] >= cutoff) &
    (orders["order_ts"] < pd.Timestamp("2025-10-01"))
]

features = before.groupby("customer_id").agg(
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum"),
    avg_order_value=("order_amount", "mean"),
    avg_discount=("discount_pct", "mean")
).reset_index()

features["last_order"] = before.groupby("customer_id")["order_ts"].max().values
features["recency"] = (cutoff - features["last_order"]).dt.days
features = features.drop(columns=["last_order"])

features = features.merge(customers, on="customer_id", how="left")

active_ids = set(after["customer_id"])
features["lapsed"] = (~features["customer_id"].isin(active_ids)).astype(int)

print("COHORT:", len(features))
print("LAPSED:", features["lapsed"].sum())
print("LAPSE RATE:", round(features["lapsed"].mean() * 100, 2), "%")

X = features.drop(columns=["customer_id", "lapsed"])
y = features["lapsed"]

numeric = X.select_dtypes(include="number").columns.tolist()
categorical = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression tuning
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

lr_grid = GridSearchCV(
    lr_pipeline,
    {"model__C": [0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="f1"
)

lr_grid.fit(X_train, y_train)
lr_pred = lr_grid.predict(X_test)

print("\nLOGISTIC REGRESSION")
print("Best Parameters:", lr_grid.best_params_)
print(classification_report(y_test, lr_pred, zero_division=0))
print("F1 SCORE:", round(f1_score(y_test, lr_pred), 3))

# Random Forest tuning with class balancing
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

rf_grid = GridSearchCV(
    rf_pipeline,
    {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20]
    },
    cv=5,
    scoring="f1"
)

rf_grid.fit(X_train, y_train)
rf_pred = rf_grid.predict(X_test)

print("\nRANDOM FOREST")
print("Best Parameters:", rf_grid.best_params_)
print(classification_report(y_test, rf_pred, zero_division=0))
print("F1 SCORE:", round(f1_score(y_test, rf_pred), 3))
print("LAPSED CUSTOMER RECALL:", round(recall_score(y_test, rf_pred), 3))

# Save the best model
best_model = rf_grid.best_estimator_
joblib.dump(best_model, "outputs/lapse_model.joblib")

print("\nSaved: outputs/lapse_model.joblib")
print("LAPSE PREDICTION COMPLETED")