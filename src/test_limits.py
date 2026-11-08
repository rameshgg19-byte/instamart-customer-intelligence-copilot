import refund_agent as agent
import mock_llm

message = "My order ORD-100048 reached me very late. I would like a refund."

# TEST 1: Token budget
print("\n========== TOKEN BUDGET TEST ==========")

mock_llm.reset()

result = agent.run_agent(
    message,
    max_tokens=1
)

print("RESULT:", result["result"])
print("TOKENS:", result["tokens"])

if result["result"] == "FALLBACK_TOKEN_BUDGET":
    print("TOKEN BUDGET TEST PASSED")
else:
    print("TOKEN BUDGET TEST FAILED")


# TEST 2: Timeout
print("\n========== TIMEOUT TEST ==========")

mock_llm.reset()

result = agent.run_agent(
    message,
    timeout_seconds=0
)

print("RESULT:", result["result"])

if result["result"] == "FALLBACK_TIMEOUT":
    print("TIMEOUT TEST PASSED")
else:
    print("TIMEOUT TEST FAILED")