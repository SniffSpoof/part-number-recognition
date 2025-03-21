import pandas as pd
import glob
import re

# File paths
file_pattern = "/content/predicted_data_*.xlsx"
files = glob.glob(file_pattern)

filtered_files = [file for file in files if re.match(r".*predicted_data_\d+\.xlsx$", file)]

all_data = pd.concat([pd.read_excel(file) for file in filtered_files], ignore_index=True)

duplicates = all_data[all_data.duplicated(subset="url", keep=False)]

# Save the merged data and duplicates to separate files
all_data.to_excel("/content/merged_data.xlsx", index=False)
duplicates.to_excel("/content/duplicates.xlsx", index=False)

print(f"Merged data saved: /content/merged_data.xlsx")
df = pd.read_excel('/content/merged_data.xlsx')
df_q = df.query("predicted_number != 'NONE'").url
print(f'{len(df_q)} of {len(df)} are predicted with numbers')
[print(i) for i in enumerate(df_q)]
df_q1 = df.query("predicted_number == 'NONE'").url
print(f'{len(df_q1)} of {len(df)} are predicted with NONE')
[print(i) for i in enumerate(df_q1)]
df.head(100)

print(f"Dublicates saved: /content/duplicates.xlsx")
df = pd.read_excel('/content/duplicates.xlsx')
df_q = df.query("predicted_number != 'NONE'").url
print(f'{len(df_q)} of {len(df)} are predicted with numbers')
[print(i) for i in enumerate(df_q)]
df_q1 = df.query("predicted_number == 'NONE'").url
print(f'{len(df_q1)} of {len(df)} are predicted with NONE')
[print(i) for i in enumerate(df_q1)]
df.head(100)
