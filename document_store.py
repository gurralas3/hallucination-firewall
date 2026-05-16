import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

INDEXES_DIR = "org_indexes"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def _chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c.strip()) > 50]

def _read_file(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def build_index(org_id: str, file_paths: list[str]):
    all_chunks = []
    for path in file_paths:
        text = _read_file(path)
        all_chunks.extend(_chunk_text(text))

    embeddings = embedder.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    org_dir = os.path.join(INDEXES_DIR, org_id)
    os.makedirs(org_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(org_dir, "index.faiss"))
    with open(os.path.join(org_dir, "chunks.json"), "w") as f:
        json.dump(all_chunks, f)

    print(f"Index built for '{org_id}' with {len(all_chunks)} chunks.")

# Squared L2 distance above this = no sufficiently relevant WHO record found.
# all-MiniLM-L6-v2 embeddings are normalized, so squared L2 = 2 - 2*cos_sim.
# dist=1.0 corresponds to cos_sim=0.5 — a weak match.
RETRIEVAL_THRESHOLD = 1.0


def search(query: str, org_id: str, top_k: int = 3) -> list[str]:
    org_dir = os.path.join(INDEXES_DIR, org_id)
    index = faiss.read_index(os.path.join(org_dir, "index.faiss"))
    with open(os.path.join(org_dir, "chunks.json")) as f:
        chunks = json.load(f)

    query_vec = embedder.encode([query]).astype("float32")
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def search_with_scores(query: str, org_id: str, top_k: int = 3) -> list[tuple]:
    """Return (chunk_text, l2_distance, similarity_pct) triples.

    similarity_pct = max(0, (1 - dist/2) * 100)
    Lower L2 distance = higher similarity. At dist=0 → 100%, dist=2 → 0%.
    """
    org_dir = os.path.join(INDEXES_DIR, org_id)
    index = faiss.read_index(os.path.join(org_dir, "index.faiss"))
    with open(os.path.join(org_dir, "chunks.json")) as f:
        chunks = json.load(f)

    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            sim_pct = round(max(0.0, (1.0 - float(dist) / 2.0)) * 100, 1)
            results.append((chunks[idx], float(dist), sim_pct))
    return results


if __name__ == "__main__":
    # Create a sample document and test indexing + search
    sample_doc = "sample_docs/company_policy.txt"
    os.makedirs("sample_docs", exist_ok=True)

    with open(sample_doc, "w") as f:
        f.write("""
        COMPANY POLICY DOCUMENT — TechCorp Inc.

        Refund Policy:
        All sales are final. We do not offer refunds under any circumstances.
        Customers may exchange defective products within 7 days of purchase with proof of receipt.

        Shipping Policy:
        Standard shipping takes 5-7 business days. Express shipping takes 2 business days at an additional cost of $15.
        We ship to the United States only. International shipping is not available.

        Customer Support:
        Support is available Monday to Friday, 9am to 5pm EST.
        Email: support@techcorp.com. Response time is within 24 hours on business days.
        """)

    build_index("techcorp", [sample_doc])

    print("\n--- Search Test ---")
    results = search("what is the refund policy", "techcorp")
    for i, chunk in enumerate(results, 1):
        print(f"\nResult {i}:\n{chunk[:200]}...")
