# Instamart Customer Intelligence & Support Copilot

## 1. Overview

The Instamart Customer Intelligence & Support Copilot is an AI-based customer analytics and support automation project.

The project analyzes customer data, identifies customer segments, predicts customer lapses, classifies support tickets, and automates refund-related customer support activities.

It uses Python, Pandas, Scikit-learn, SQLite, TF-IDF, and a mock LLM to demonstrate an offline AI-powered customer support system.

## 2. Reproduce

### Requirements
- Python 3.11
- VS Code
- Pandas
- NumPy
- Scikit-learn
- Joblib
- SQLite

### Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the project data:

```bash
python generate_data.py
```

Run the project scripts from the project root directory.

## 3. Part A – Data Audit and Cleaning

This part focuses on understanding, cleaning, and validating customer, product, and order datasets.

### Activities
- Identified missing values and duplicate records.
- Detected invalid and inconsistent data.
- Standardized city and gender values.
- Handled missing customer ages.
- Removed duplicate orders.
- Converted order amounts into numeric values.
- Handled invalid delivery times.
- Stored cleaned data in SQLite.

### Outputs
- `data/clean/customers.csv`
- `data/clean/products.csv`
- `data/clean/orders.csv`
- `data/instamart.db`
- `sql/queries.sql`

## 4. Part B – Customer Intelligence

This part analyzes customer purchasing behavior and identifies important customer groups.

### Activities
- Calculated Recency, Frequency, and Monetary (RFM) values.
- Applied K-Means clustering for customer segmentation.
- Developed a customer lapse prediction model.
- Identified customers who may need retention campaigns.

### Outputs
- `outputs/customer_segments.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## 5. Part C – Customer Ticket Triage

This part automates customer support ticket processing.

### Activities
- Validated customer support tickets.
- Detected empty and excessively long messages.
- Redacted sensitive information such as email addresses, phone numbers, and order IDs.
- Classified customer tickets using zero-shot and few-shot prompting.
- Implemented retry handling and JSON validation.
- Used keyword-based fallback classification.
- Applied token budget controls.
- Routed tickets based on priority and category.

### Outputs
- `outputs/zero_shot_results.csv`
- `outputs/few_shot_results.csv`
- `outputs/retry_logs.csv`
- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`

## 6. Part D – RAG Assistant and Refund Agent

### A. RAG Policy Assistant

The Retrieval-Augmented Generation (RAG) assistant answers customer questions using company policy documents.

Activities:
- Loaded five policy documents.
- Divided documents into smaller text chunks.
- Used TF-IDF and cosine similarity for document retrieval.
- Retrieved relevant policy information.
- Generated answers using the mock LLM.
- Evaluated answers against policy questions.
- Tested retrieval with and without chunk overlap.

### B. ReAct Refund Agent

The refund agent automates refund processing using multiple tools.

Tools used:
- `get_order` – Retrieves order information from SQLite.
- `search_policy` – Searches refund policies.
- `calculate_refund` – Calculates eligible refund amounts.
- `issue_refund` – Records approved refunds.
- `escalate_to_human` – Sends high-value refunds for human approval.

### Refund Rules
- Only completed orders are eligible for refunds.
- Delivery taking more than 45 minutes qualifies for a full refund.
- Refunds up to INR 1000 can be approved automatically.
- Refunds above INR 1000 require human approval.
- Cancelled orders are not eligible for refunds.

### C. Agent Reliability

The agent includes:
- Maximum iteration control.
- Token budget protection.
- Timeout handling.
- Checkpoint saving.
- Crash recovery.
- JSONL execution logging.
- Refund transaction logging.

### D. ReAct vs Plan-Execute Agent Comparison

**ReAct Approach**

ReAct stands for Reasoning and Acting. The agent selects a tool, executes it, observes the result, and decides the next action.

In this project, the refund agent uses ReAct to:
1. Retrieve customer order details.
2. Search the refund policy.
3. Calculate the eligible refund amount.
4. Issue a refund or escalate the request to a human.

**Plan-Execute Approach**

Plan-Execute first creates a complete plan for solving the task. After creating the plan, the agent executes each step in order. Replanning may be required if unexpected results occur.

**Comparison**

| ReAct | Plan-Execute |
|---|---|
| Decides actions step by step. | Creates a plan before execution. |
| Uses observations to decide the next action. | Follows the prepared plan. |
| Flexible when results change. | Suitable for predictable tasks. |
| May require more model calls. | Can reduce repeated planning calls. |
| Adapts to unexpected results. | May require replanning when errors occur. |

**Conclusion**

This project uses the ReAct approach because refund processing requires checking order details, validating policies, calculating refund amounts, and deciding whether to issue or escalate a refund.

### Outputs
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `state/`

## 7. Limitations and Next Steps

### Limitations
- The project uses a mock LLM instead of a live commercial LLM API.
- The datasets are synthetic.
- Policy retrieval uses TF-IDF instead of advanced embedding models.
- Refund processing is demonstrated using a local database and JSONL ledger.
- The system does not connect to a real Instamart service.

### Future Improvements
- Integrate a real LLM API.
- Use advanced semantic search and embeddings.
- Develop a web-based customer support dashboard.
- Improve prediction accuracy with larger datasets.
- Integrate real-time order tracking.
- Add secure payment and refund integrations.

## Conclusion

The Instamart Customer Intelligence & Support Copilot demonstrates how data analytics, machine learning, natural language processing, RAG, and AI agents can work together to improve customer intelligence and automate customer support operations.