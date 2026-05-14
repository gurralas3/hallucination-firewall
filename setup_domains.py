"""Index legal and finance domains into FAISS."""
import os

def setup_all():
    legal_index   = os.path.join("org_indexes", "legal",   "index.faiss")
    finance_index = os.path.join("org_indexes", "finance", "index.faiss")

    if not os.path.exists(legal_index):
        print("Building legal index...")
        from legal_processor import process_and_index as legal_index_fn
        legal_index_fn()
    else:
        print("[legal] Index already exists — skipping.")

    if not os.path.exists(finance_index):
        print("Building finance index...")
        from finance_processor import process_and_index as finance_index_fn
        finance_index_fn()
    else:
        print("[finance] Index already exists — skipping.")

if __name__ == "__main__":
    setup_all()
    print("\nAll domain indexes ready.")
