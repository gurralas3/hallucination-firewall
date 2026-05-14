import pandas as pd
import os
from document_store import build_index

DATASET_PATH = "sample_docs/healthcare_dataset.csv"
OUTPUT_DIR   = "sample_docs/hospital_records"
SAMPLE_SIZE  = 500

def normalize_name(name: str) -> str:
    return " ".join(word.capitalize() for word in name.strip().split())

def record_to_text(row) -> str:
    return (
        f"Patient: {normalize_name(row['Name'])}. "
        f"Age: {row['Age']}. "
        f"Gender: {row['Gender']}. "
        f"Blood Type: {row['Blood Type']}. "
        f"Medical Condition: {row['Medical Condition']}. "
        f"Date of Admission: {row['Date of Admission']}. "
        f"Discharge Date: {row['Discharge Date']}. "
        f"Admission Type: {row['Admission Type']}. "
        f"Doctor: {row['Doctor']}. "
        f"Hospital: {row['Hospital']}. "
        f"Insurance Provider: {row['Insurance Provider']}. "
        f"Room Number: {row['Room Number']}. "
        f"Medication: {row['Medication']}. "
        f"Test Results: {row['Test Results']}. "
        f"Billing Amount: ${row['Billing Amount']:.2f}."
    )

def process_and_index():
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(df_sample)} records from {len(df)} total.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    chunk_files = []
    for i, (_, row) in enumerate(df_sample.iterrows()):
        file_path = os.path.join(OUTPUT_DIR, f"record_{i:04d}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(record_to_text(row))
        chunk_files.append(file_path)

    print(f"Written {len(chunk_files)} individual record files. Building FAISS index...")
    build_index("hospital", chunk_files)
    print("Hospital knowledge base ready.")

    print("\nSample records indexed:")
    for _, row in df_sample.head(3).iterrows():
        print(f"  - {normalize_name(row['Name'])}, {row['Age']}yo, {row['Medical Condition']}, on {row['Medication']}")

if __name__ == "__main__":
    process_and_index()
