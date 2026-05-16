import os
from document_store import build_index

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs", "who_medicines.txt")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs", "chw_records_indexed")

def process():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    medicines = [m.strip() for m in content.split("\n\n") if m.strip().startswith("MEDICINE:")]
    for i, record in enumerate(medicines):
        path = os.path.join(OUTPUT_DIR, f"medicine_{i:03d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(record)

    print(f"Indexed {len(medicines)} WHO Essential Medicines records.")
    files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt")]
    build_index("chw", files)

if __name__ == "__main__":
    process()
