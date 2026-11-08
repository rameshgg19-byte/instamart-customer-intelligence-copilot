import refund_agent as agent
import mock_llm

RUN_ID = "recovery_test_s1"

message = "My order ORD-100048 reached me very late. I would like a refund."

# Prevent this test from adding another entry to the real refund ledger
agent.TOOL_MAP["issue_refund"] = lambda order_id, refund_inr: {
    "order_id": order_id,
    "refund_inr": refund_inr,
    "status": "TEST_ONLY"
}

mock_llm.reset()

print("\nSTARTING AGENT")

try:
    agent.run_agent(
        message,
        max_iterations=6,
        run_id=RUN_ID,
        crash_after_tools=2
    )

except RuntimeError as error:
    print("\nSIMULATED CRASH:", error)

print("\nCHECKPOINT CREATED")

print("\nRESUMING AGENT")

mock_llm.reset()

result = agent.run_agent(
    message,
    max_iterations=6,
    run_id=RUN_ID,
    resume=True
)

print("\nFINAL RESULT:", result["result"])
print("TOTAL TOKENS:", result["tokens"])
print("TOTAL STEPS:", result["steps"])

if result["result"].startswith("REFUND"):
    print("\nCRASH RECOVERY TEST PASSED")
else:
    print("\nCRASH RECOVERY TEST FAILED")