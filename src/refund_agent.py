import sqlite3
import json
import pandas as pd
import sys
from pathlib import Path
import uuid
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_llm

DB_FILE = ROOT / "data" / "instamart.db"
SCENARIO_FILE = ROOT / "data" / "raw" / "agent_scenarios.csv"


def get_order(order_id):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute(
        "SELECT * FROM orders WHERE order_id = ?",
        (order_id,)
    )

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {"error": "Order not found"}

    return dict(row)


def search_policy(query):
    from rag_assistant import (
        load_documents,
        create_chunks,
        retrieve
    )

    from sklearn.feature_extraction.text import TfidfVectorizer

    documents = load_documents()
    chunks = create_chunks(documents)

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )

    results = retrieve(
        query,
        chunks,
        vectorizer,
        matrix
    )

    return {
        "results": [
            {
                "source": item["source"],
                "text": item["text"]
            }
            for item in results
        ]
    }


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None:
        refund = 0
    elif delivery_minutes > 45:
        refund = float(order_amount)
    else:
        refund = 0

    return {"refund_inr": round(refund, 2)}


def issue_refund(order_id, refund_inr):
    ledger_file = ROOT / "outputs" / "refund_ledger.jsonl"
    ledger_file.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "order_id": order_id,
        "refund_inr": refund_inr,
        "status": "ISSUED"
    }

    with open(ledger_file, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return record


def escalate_to_human(order_id, refund_inr):
    return {
        "order_id": order_id,
        "refund_inr": refund_inr,
        "status": "HUMAN_APPROVAL_REQUIRED"
    }


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


def run_agent(
    user_message,
    max_iterations=6,
    run_id=None,
    resume=False,
    crash_after_tools=None,
    max_tokens=3000,
    timeout_seconds=30
):

    state_dir = ROOT / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        run_id = uuid.uuid4().hex

    checkpoint_file = state_dir / f"{run_id}.json"

    start_time = time.time()

    if resume:

        with open(checkpoint_file, "r", encoding="utf-8") as file:
            checkpoint = json.load(file)

        messages = checkpoint["messages"]
        total_tokens = checkpoint["tokens"]
        start_step = checkpoint["next_step"]
        tool_count = checkpoint["tool_count"]
        user_message = checkpoint["user_message"]

        print("RESUMING RUN:", run_id)
        print("TOOLS ALREADY COMPLETED:", tool_count)

    else:

        messages = [
            {"role": "system", "content": "AGENT"},
            {"role": "user", "content": user_message}
        ]

        total_tokens = 0
        start_step = 0
        tool_count = 0

    for step in range(start_step, max_iterations):

        if time.time() - start_time > timeout_seconds:
            return {
                "result": "FALLBACK_TIMEOUT",
                "tokens": total_tokens,
                "steps": step,
                "run_id": run_id
            }

        response = mock_llm.chat(messages)

        total_tokens += response.usage.total_tokens

        if total_tokens > max_tokens:
            return {
                "result": "FALLBACK_TOKEN_BUDGET",
                "tokens": total_tokens,
                "steps": step + 1,
                "run_id": run_id
            }

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls

        if not tool_calls:
            return {
                "result": assistant_message.content,
                "tokens": total_tokens,
                "steps": step + 1,
                "run_id": run_id
            }

        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments
                }
                for call in tool_calls
            ]
        })

        for call in tool_calls:

            if time.time() - start_time >= timeout_seconds:
                return {
                    "result": "FALLBACK_TIMEOUT",
                    "tokens": total_tokens,
                    "steps": step + 1,
                    "run_id": run_id
                }

            tool_name = call.name
            arguments = call.arguments

            print("STEP:", step + 1)
            print("TOOL:", tool_name)
            print("ARGUMENTS:", arguments)

            tool_function = TOOL_MAP[tool_name]
            tool_result = tool_function(**arguments)

            print("RESULT:", tool_result)
            print("-" * 40)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": json.dumps(tool_result)
            })

            tool_count += 1

            checkpoint = {
                "run_id": run_id,
                "user_message": user_message,
                "messages": messages,
                "tokens": total_tokens,
                "next_step": step + 1,
                "tool_count": tool_count
            }

            with open(checkpoint_file, "w", encoding="utf-8") as file:
                json.dump(checkpoint, file, indent=4)

            print("CHECKPOINT SAVED:", checkpoint_file)

            if crash_after_tools == tool_count:
                raise RuntimeError("SIMULATED AGENT CRASH")

    return {
        "result": "FALLBACK_MAX_ITERATIONS",
        "tokens": total_tokens,
        "steps": max_iterations,
        "run_id": run_id
    }


def main():

    scenarios = pd.read_csv(SCENARIO_FILE)

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    ledger_file = output_dir / "refund_ledger.jsonl"
    log_file = output_dir / "agent_runs.jsonl"

    # Start fresh for this test batch
    ledger_file.write_text("", encoding="utf-8")
    log_file.write_text("", encoding="utf-8")

    print("TOTAL SCENARIOS:", len(scenarios))

    for _, scenario in scenarios.iterrows():

        scenario_id = scenario["scenario_id"]
        message = scenario["message"]
        expected = scenario["expected_outcome"]

        print("\n" + "=" * 45)
        print("RUNNING:", scenario_id)
        print("MESSAGE:", message)

        mock_llm.reset()

        result = run_agent(message, max_iterations=6)

        actual = result["result"]

        passed = actual.startswith(expected)

        record = {
            "scenario_id": scenario_id,
            "expected": expected,
            "actual": actual,
            "tokens": result["tokens"],
            "steps": result["steps"],
            "passed": passed
        }

        with open(log_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        print("\nEXPECTED:", expected)
        print("ACTUAL:", actual)
        print("TOKENS:", result["tokens"])
        print("STEPS:", result["steps"])
        print("STATUS:", "PASS" if passed else "FAIL")

    print("\n" + "=" * 45)
    print("ALL SCENARIOS COMPLETED")
    print("AGENT LOG:", log_file)
    print("REFUND LEDGER:", ledger_file)


if __name__ == "__main__":
    main()