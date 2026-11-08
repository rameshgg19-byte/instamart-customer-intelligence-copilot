import pandas as pd
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import mock_llm

tickets = pd.read_csv("data/raw/tickets.csv")

MAX_INPUT_TOKENS = 400

patterns = {
    "email": (r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
    "phone": (r"\b[6-9]\d{9}\b", "[PHONE]"),
    "order_id": (r"ORD-\d{6}", "[ORDER_ID]")
}

redaction_counts = {
    "email": 0,
    "phone": 0,
    "order_id": 0
}

results = []

for _, ticket in tickets.iterrows():

    original_text = ticket["text"]

    if pd.isna(original_text) or not str(original_text).strip():
        results.append("skipped_empty")
        continue

    text = str(original_text)

    for name, (pattern, placeholder) in patterns.items():
        text, count = re.subn(pattern, placeholder, text)
        redaction_counts[name] += count

    for pattern, placeholder in patterns.values():
        assert not re.search(pattern, text)

    if mock_llm.est_tokens(str(original_text)) > MAX_INPUT_TOKENS:
        results.append("flagged_long")
        continue

    results.append("ready")

print("TOTAL TICKETS:", len(tickets))
print("EMPTY:", results.count("skipped_empty"))
print("LONG:", results.count("flagged_long"))
print("READY:", results.count("ready"))

print("\nREDACTION COUNTS:")
print(redaction_counts)

print("TOTAL REDACTIONS:", sum(redaction_counts.values()))

print("\nALL REDACTION CHECKS PASSED")