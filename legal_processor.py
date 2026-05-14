import pandas as pd
import os
from document_store import build_index

DATASET_PATH = "sample_docs/supreme_court.csv"
OUTPUT_DIR   = "sample_docs/legal_records_indexed"
SAMPLE_SIZE  = 200

DISPOSITION_MAP = {
    1: "Stay/Petition Granted", 2: "Affirmed", 3: "Reversed",
    4: "Reversed and Remanded", 5: "Vacated and Remanded",
    6: "Affirmed and Reversed in Part", 7: "Dismissed", 8: "Certiorari Dismissed"
}
DIRECTION_MAP = {1: "Conservative", 2: "Liberal", 3: "Unspecifiable"}

def record_to_text(row) -> str:
    disposition = DISPOSITION_MAP.get(int(row["case_disposition"]) if pd.notna(row["case_disposition"]) else 0, "Unknown")
    direction   = DIRECTION_MAP.get(int(row["decision_direction"]) if pd.notna(row["decision_direction"]) else 0, "Unknown")
    majority    = int(row["majority_votes"]) if pd.notna(row["majority_votes"]) else "Unknown"
    minority    = int(row["minority_votes"]) if pd.notna(row["minority_votes"]) else "Unknown"
    return (
        f"Case: {row['case_name']}. "
        f"Term: {row['term']}. "
        f"Date Decided: {row['date_decision']}. "
        f"Chief Justice: {row['chief_justice']}. "
        f"Majority Votes: {majority}. "
        f"Minority Votes: {minority}. "
        f"Outcome: {disposition}. "
        f"Decision Direction: {direction}. "
        f"US Citation: {row['us_citation']}."
    )

def process_and_index():
    print(f"Loading legal dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df = df.dropna(subset=["case_name", "date_decision", "chief_justice"])
    df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)
    print(f"Sampled {len(df_sample)} records.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    chunk_files = []
    for i, (_, row) in enumerate(df_sample.iterrows()):
        file_path = os.path.join(OUTPUT_DIR, f"record_{i:04d}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(record_to_text(row))
        chunk_files.append(file_path)

    print(f"Written {len(chunk_files)} files. Building FAISS index...")
    build_index("legal", chunk_files)
    print("Legal knowledge base ready.")

    print("\nSample records:")
    for _, row in df_sample.head(3).iterrows():
        print(f"  - {row['case_name'][:60]}, {row['date_decision']}, Chief Justice: {row['chief_justice']}")

if __name__ == "__main__":
    process_and_index()
