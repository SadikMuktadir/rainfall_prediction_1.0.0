import pandas as pd

# Load Excel file
file_path = "../data/last-7-days-bamis-rainfall-data.xlsx"
df = pd.read_excel(file_path)

# Replace blank values with average
column_name = "Rainfall Total (mm)"

avg_value = df[column_name].mean()

df[column_name] = df[column_name].fillna(avg_value)

# Save new file
df.to_excel("updated_file-new.xlsx", index=False)

print("Done!")