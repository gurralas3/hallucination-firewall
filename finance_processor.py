import pandas as pd
import os
from document_store import build_index

DATASET_PATH = "sample_docs/bank_customers.csv"
OUTPUT_DIR   = "sample_docs/finance_records_indexed"
SAMPLE_SIZE  = 200

def record_to_text(row) -> str:
    return (
        f"Client: {row['Name']}. "
        f"Age: {row['Age']}. "
        f"Gender: {row['Gender']}. "
        f"Account Type: {row['Account Type']}. "
        f"Account Balance: ${float(row['Account Balance']):,.2f}. "
        f"Registration Date: {row['Registration Date']}. "
        f"Address: {str(row['Address']).splitlines()[0]}. "
        f"Phone: {row['Phone Number']}."
    )

def process_and_index():
    print(f"Loading finance dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
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
    build_index("finance", chunk_files)
    print("Finance knowledge base ready.")

    print("\nSample records:")
    for _, row in df_sample.head(3).iterrows():
        print(f"  - {row['Name']}, {row['Account Type']}, ${float(row['Account Balance']):,.2f}")

if __name__ == "__main__":
    process_and_index()
