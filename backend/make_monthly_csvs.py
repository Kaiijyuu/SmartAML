import pandas as pd

df = pd.read_csv("all_transactions.csv")
df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

months = {
    "jan_2025_transactions.csv": ("2025-01-01", "2025-01-31"),
    "feb_2025_transactions.csv": ("2025-02-01", "2025-02-28"),
    "mar_2025_transactions.csv": ("2025-03-01", "2025-03-31"),
}

for filename, (start, end) in months.items():
    month_df = df[(df["dt"] >= start) & (df["dt"] <= end)].copy()
    month_df.to_csv(filename, index=False)
    print(filename, len(month_df), "rows created")