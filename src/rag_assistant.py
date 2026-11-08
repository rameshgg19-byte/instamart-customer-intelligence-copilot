import pandas as pd
import re
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_llm

POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"

CHUNK_SIZE = 40
OVERLAP = 10


def load_documents():
    documents = {}

    for file in POLICY_DIR.glob("*.txt"):
        documents[file.name] = file.read_text(encoding="utf-8")

    return documents


def create_chunks(documents, overlap=OVERLAP):
    chunks = []

    for filename, text in documents.items():
        words = text.split()
        step = CHUNK_SIZE - overlap
        start = 0

        while start < len(words):
            chunk_words = words[start:start + CHUNK_SIZE]

            chunks.append({
                "source": filename,
                "text": " ".join(chunk_words)
            })

            if start + CHUNK_SIZE >= len(words):
                break

            start += step

    return chunks


def retrieve(question, chunks, vectorizer, matrix, top_k=3):
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, matrix)[0]

    indexes = scores.argsort()[::-1][:top_k]

    results = []

    for i in indexes:
        results.append({
            "source": chunks[i]["source"],
            "text": chunks[i]["text"],
            "score": scores[i]
        })

    return results


def answer_question(question, results):
    context = "\n".join(
        f"[{item['source']}] {item['text']}"
        for item in results
    )

    prompt = f"""
CONTEXT:
{context}

QUESTION:
{question}
"""

    messages = [
        {
            "role": "system",
            "content": "POLICY ASSISTANT"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = mock_llm.chat(messages)

    return response.choices[0].message.content


def main():
    documents = load_documents()
    chunks = create_chunks(documents)

    print("TOTAL POLICY DOCUMENTS:", len(documents))
    print("TOTAL CHUNKS:", len(chunks))

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )

    print("\nRETRIEVAL TEST:")

    question = "What delivery time makes a completed order eligible for a full refund?"

    results = retrieve(question, chunks, vectorizer, matrix)

    for item in results:
        print(
            "Source:", item["source"],
            "Score:", round(item["score"], 4)
        )

    print("\nANSWER TEST:")

    response = answer_question(question, results)
    print("Answer:", response)

    print("\nPOLICY EVALUATION:")

    evaluation = pd.read_csv(EVAL_FILE)

    expected_doc = 0
    expected_fact = 0
    fallback = 0

    for _, row in evaluation.iterrows():

        results = retrieve(
            row["question"],
            chunks,
            vectorizer,
            matrix
        )

        sources = [item["source"] for item in results]

        context = " ".join(
            item["text"] for item in results
        )

        if row["expected_doc"] in sources:
            expected_doc += 1

        if str(row["expected_fact"]).lower() in context.lower():
            expected_fact += 1

        response = answer_question(row["question"], results)

        if response.strip().lower().startswith("i could not find"):
            fallback += 1

    print("TOTAL QUESTIONS:", len(evaluation))
    print("EXPECTED DOCUMENT FOUND:", expected_doc, "/", len(evaluation))
    print("EXPECTED FACT FOUND:", expected_fact, "/", len(evaluation))
    print("FALLBACK ANSWERS:", fallback)

    print("\nZERO OVERLAP COMPARISON:")

    zero_chunks = create_chunks(documents, overlap=0)

    zero_vectorizer = TfidfVectorizer()

    zero_matrix = zero_vectorizer.fit_transform(
        [chunk["text"] for chunk in zero_chunks]
    )

    failures = []

    for _, row in evaluation.iterrows():

        results = retrieve(
            row["question"],
            zero_chunks,
            zero_vectorizer,
            zero_matrix
        )

        context = " ".join(
            item["text"] for item in results
        )

        if str(row["expected_fact"]).lower() not in context.lower():
            failures.append(row["question"])

    print("ZERO OVERLAP CHUNKS:", len(zero_chunks))
    print("ZERO OVERLAP FAILURES:", len(failures))

    for question in failures:
        print("Failed question:", question)

    print("\nRAG POLICY ASSISTANT COMPLETED")


if __name__ == "__main__":
    main()
