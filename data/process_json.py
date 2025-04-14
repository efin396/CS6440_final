import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # Optional, for progress bar
from pandas import json_normalize

json_dir = Path("/Users/willferguson/Downloads/GT Spring 2025/CS 6440/CS6440Project/data/output_500/fhir").resolve()
print(f"Resolved path: {json_dir}")
print(f"Exists? {json_dir.exists()}")
print(f"Is directory? {json_dir.is_dir()}")

# for file in json_dir.iterdir():
#     print(repr(file.name))

json_files = list(json_dir.glob("*"))

records = []

for file_path in tqdm(json_files, desc="Processing FHIR files"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)['entry']
        types = [r['resource']['resourceType'] for r in data]
        # flat = json_normalize(data)  # flatten the FHIR JSON structure
        records.append(data)

df = pd.concat(records, ignore_index=True)
df.to_csv("fhir_dataset.csv", index=False)

print(f"Saved CSV with {len(df)} rows and {len(df.columns)} columns.")