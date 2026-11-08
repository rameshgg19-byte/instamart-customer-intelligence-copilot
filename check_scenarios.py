import pandas as pd

data = pd.read_csv("data/raw/agent_scenarios.csv")

print("COLUMNS:")
print(data.columns.tolist())

print("\nALL SCENARIOS:")
print(data.to_string(index=False))