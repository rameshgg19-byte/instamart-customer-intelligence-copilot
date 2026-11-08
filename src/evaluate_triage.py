import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent.parent

tickets = pd.read_csv(
    ROOT / "data/raw/tickets.csv",
    keep_default_na=False
)

categories = ["billing", "delivery", "technical", "fraud"]

def evaluate(filename, prompt_name):

    results = pd.read_csv(ROOT / "outputs" / filename)

    data = results.merge(
        tickets[["ticket_id", "true_category"]],
        on="ticket_id"
    )

    classified = data[
        data["status"].isin(["classified_json", "classified_keyword"])
    ].copy()

    actual = classified["true_category"]
    predicted = classified["category"]

    accuracy = accuracy_score(actual, predicted)

    total_tokens = results["tokens_used"].sum()
    classified_count = len(classified)

    print("\n------------------------------")
    print("PROMPT:", prompt_name)
    print("------------------------------")

    print("Total tickets:", len(results))
    print("Classified tickets:", classified_count)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Total tokens:", total_tokens)
    print(
        "Tokens per classified ticket:",
        round(total_tokens / classified_count, 2)
    )

    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(actual, predicted, labels=categories))

    print("\nCLASSIFICATION REPORT:")
    print(
        classification_report(
            actual,
            predicted,
            labels=categories,
            zero_division=0
        )
    )


evaluate("zero_shot_results.csv", "ZERO-SHOT")

evaluate("few_shot_results.csv", "FEW-SHOT")