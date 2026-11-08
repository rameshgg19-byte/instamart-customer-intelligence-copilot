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

MAX_INPUT_TOKENS = 400

patterns = {
    "email": (r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
    "phone": (r"\b[6-9]\d{9}\b", "[PHONE]"),
    "order_id": (r"ORD-\d{6}", "[ORDER_ID]")
}

def prepare_tickets():

    prepared = []

    for _, ticket in tickets.iterrows():

        text = str(ticket["text"])

        if not text.strip():
            continue

        if mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            continue

        for pattern, replacement in patterns.values():
            text = re.sub(pattern, replacement, text)

        prepared.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "text": text
        })

    return prepared


ZERO_SHOT_PROMPT = """
You are a customer support ticket classifier.

CLASSIFY each customer message into exactly one category:
billing, delivery, technical, fraud.

Return JSON only in this format:
{"category":"billing","reason":"short reason"}

Do not return any extra text.
"""

FEW_SHOT_PROMPT = """
You are a customer support ticket classifier.

CLASSIFY each customer message into exactly one category:
billing, delivery, technical, fraud.

EXAMPLES:

Message: I was charged twice for my order.
Output: {"category":"billing","reason":"duplicate charge"}

Message: My account was hacked and money was stolen.
Output: {"category":"fraud","reason":"unauthorized activity"}

Message: The app crashes when I open it.
Output: {"category":"technical","reason":"app problem"}

Return JSON only in this format:
{"category":"billing","reason":"short reason"}

Do not return any extra text.
"""

BASE_DELAY = 0.05
MAX_RETRIES = 3

def classify_batch(prompt, prompt_name):

    mock_llm.reset()

    results = []
    retry_logs = []

    for ticket in prepare_tickets():

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": ticket["text"]}
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

                retry_logs.append({
                    "ticket_id": ticket["ticket_id"],
                    "retry_number": attempt + 1,
                    "wait_seconds": wait
                })

                print(
                    "Retry:",
                    ticket["ticket_id"],
                    "Attempt:",
                    attempt + 1,
                    "Wait:",
                    round(wait, 4)
                )

                time.sleep(wait)

        if response is None:
            results.append({
                "ticket_id": ticket["ticket_id"],
                "customer_id": ticket["customer_id"],
                "status": "flagged_invalid_output",
                "category": None,
                "tokens_used": 0
            })
            continue

        message = response.choices[0].message
        content = message.content or ""

        tokens = response.usage.total_tokens

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

    print("\nPROMPT:", prompt_name)
    print("TOTAL PROCESSED:", len(results))
    print("JSON:", sum(r["status"] == "classified_json" for r in results))
    print("KEYWORD:", sum(r["status"] == "classified_keyword" for r in results))
    print("INVALID:", sum(r["status"] == "flagged_invalid_output" for r in results))
    print("RETRIES:", len(retry_logs))
    print("TOTAL TOKENS:", sum(r["tokens_used"] for r in results))

    return results, retry_logs


zero_results, zero_retries = classify_batch(
    ZERO_SHOT_PROMPT,
    "ZERO_SHOT"
)

few_results, few_retries = classify_batch(
    FEW_SHOT_PROMPT,
    "FEW_SHOT"
)

output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

pd.DataFrame(zero_results).to_csv(
    output_dir / "zero_shot_results.csv",
    index=False
)

pd.DataFrame(few_results).to_csv(
    output_dir / "few_shot_results.csv",
    index=False
)

pd.DataFrame(zero_retries + few_retries).to_csv(
    output_dir / "retry_logs.csv",
    index=False
)

print("\nCLASSIFICATION FILES SAVED SUCCESSFULLY")