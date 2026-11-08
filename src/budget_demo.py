import pandas as pd
import re
import json
import sys
import time
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import mock_llm

tickets = pd.read_csv(
    ROOT / "data/raw/tickets.csv",
    keep_default_na=False
)

patterns = [
    (r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
    (r"\b[6-9]\d{9}\b", "[PHONE]"),
    (r"ORD-\d{6}", "[ORDER_ID]")
]

prompt = """
You are a customer support ticket classifier.

CLASSIFY each message into billing, delivery, technical, or fraud.

EXAMPLES:

Message: I was charged twice.
Output: {"category":"billing","reason":"duplicate charge"}

Message: My account was hacked.
Output: {"category":"fraud","reason":"unauthorized activity"}

Message: My app crashes.
Output: {"category":"technical","reason":"app problem"}

Return JSON only:
{"category":"billing","reason":"short reason"}
"""

BUDGET = 6000
MAX_RETRIES = 3
BASE_DELAY = 0.05

mock_llm.reset()

running_tokens = 0
results = []

for _, ticket in tickets.iterrows():

    text = str(ticket["text"])

    if not text.strip():
        results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "status": "skipped_empty",
            "category": None,
            "tokens_used": 0
        })
        continue

    if mock_llm.est_tokens(text) > 400:
        results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "status": "flagged_long",
            "category": None,
            "tokens_used": 0
        })
        continue

    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)

    if running_tokens >= BUDGET:
        results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "status": "budget_exhausted",
            "category": None,
            "tokens_used": 0
        })
        continue

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]

    response = None

    for attempt in range(MAX_RETRIES + 1):

        try:
            response = mock_llm.chat(messages)
            break

        except mock_llm.TransientLLMError:

            if attempt == MAX_RETRIES:
                break

            wait = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
            time.sleep(wait)

    if response is None:
        status = "flagged_invalid_output"
        category = None
        tokens = 0

    else:
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens
        running_tokens += tokens

        status = "flagged_invalid_output"
        category = None

        try:
            data = json.loads(content)

            if data.get("category") in [
                "billing", "delivery", "technical", "fraud"
            ]:
                category = data["category"]
                status = "classified_json"

        except (json.JSONDecodeError, AttributeError):

            words = re.findall(r"[a-z]+", content.lower())

            found = [
                word for word in words
                if word in ["billing", "delivery", "technical", "fraud"]
            ]

            if len(found) == 1:
                category = found[0]
                status = "classified_keyword"

    results.append({
        "ticket_id": ticket["ticket_id"],
        "customer_id": ticket["customer_id"],
        "status": status,
        "category": category,
        "tokens_used": tokens
    })

output = ROOT / "outputs"
output.mkdir(exist_ok=True)

df = pd.DataFrame(results)

df.to_csv(
    output / "triage_budget_demo.csv",
    index=False
)

print("TOTAL TICKETS:", len(df))
print("\nSTATUS COUNTS:")
print(df["status"].value_counts())

print("\nTOKEN BUDGET:", BUDGET)
print("TOKENS USED:", running_tokens)
print("BUDGET EXHAUSTED:", (df["status"] == "budget_exhausted").sum())

print("\nBUDGET DEMO SAVED SUCCESSFULLY")