import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import mock_llm

# Read original tickets
tickets = pd.read_csv(
    ROOT / "data/raw/tickets.csv",
    keep_default_na=False
)

# Read few-shot classification results
triage = pd.read_csv(
    ROOT / "outputs/few_shot_results.csv"
)

# Read customer segments
segments = pd.read_csv(
    ROOT / "outputs/customer_segments.csv"
)

# Combine original tickets and classification results
df = tickets.merge(
    triage,
    on=["ticket_id", "customer_id"],
    how="left",
    suffixes=("", "_result")
)

# Identify tickets that did not reach classification
for index, row in df.iterrows():

    if pd.isna(row["status"]):

        text = str(row["text"])

        if not text.strip():
            df.at[index, "status"] = "skipped_empty"

        elif mock_llm.est_tokens(text) > 400:
            df.at[index, "status"] = "flagged_long"

        else:
            df.at[index, "status"] = "flagged_invalid_output"

# Add customer segment information
df = df.merge(
    segments[["customer_id", "segment_name", "is_premium"]],
    on="customer_id",
    how="left"
)

# Missing customer segments are treated as non-premium
df["is_premium"] = (
    df["is_premium"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)

df["segment_name"] = df["segment_name"].fillna("Unknown")

# Assign priority and routing
def assign_priority(row):

    if row["status"] not in [
        "classified_json",
        "classified_keyword"
    ]:
        return "NORMAL"

    if row["category"] == "fraud":
        return "HIGH"

    if row["is_premium"]:
        return "HIGH"

    return "NORMAL"


def assign_route(row):

    if row["status"] not in [
        "classified_json",
        "classified_keyword"
    ]:
        return "human_review"

    if row["category"] == "fraud":
        return "security_team"

    if row["is_premium"]:
        return "premium_desk"

    return "support_queue"


df["priority"] = df.apply(assign_priority, axis=1)
df["route"] = df.apply(assign_route, axis=1)

# Select required output columns
final_df = df[
    [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used"
    ]
]

# Save final output
output_path = ROOT / "outputs" / "triage_results.csv"

final_df.to_csv(output_path, index=False)

print("TOTAL TICKETS:", len(final_df))

print("\nSTATUS COUNTS:")
print(final_df["status"].value_counts())

print("\nROUTING COUNTS:")
print(final_df["route"].value_counts())

print("\nPRIORITY COUNTS:")
print(final_df["priority"].value_counts())

print("\nFINAL FILE:", output_path)
print("FINAL TRIAGE COMPLETED SUCCESSFULLY")