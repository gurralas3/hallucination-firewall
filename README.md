# Hallucination Firewall

**Enterprise AI safety middleware that audits every LLM response against ground-truth records before it reaches the user — and corrects it silently if it's wrong.**

Built for the [Gemma 4 Good Hackathon](https://kaggle.com/competitions/gemma-4-good-hackathon) using **Gemma 4** (`gemma-4-26b-a4b-it`) via Google AI Studio.

---

## The Problem

Every organization deploying an AI assistant faces the same hidden risk: the LLM answers with confidence, but the answer is wrong. In healthcare, legal, and financial contexts, a hallucinated drug name, a fabricated case outcome, or a wrong account balance isn't a minor inconvenience — it's a liability.

Standard RAG reduces hallucinations but does not eliminate them. The LLM can still:
- Misread retrieved context and add details that aren't there
- Mix information from multiple chunks incorrectly
- Refuse to answer when it should, or answer when it shouldn't

There is no safety net between what the LLM says and what the user sees.

---

## The Solution

The Hallucination Firewall is a post-generation verification layer. It sits **between the LLM and the user**. Every response is intercepted, audited against the organization's own documents, and either passed through or silently replaced with a correct, document-grounded answer.

### Pipeline

```
User Question
      ↓
FAISS Semantic Search → Retrieve top-k relevant document chunks
      ↓
Gemma 4 generates answer using retrieved context
      ↓
┌─────────────────────────────────────────────────┐
│             HALLUCINATION FIREWALL               │
│  Gemma 4 audits answer claim-by-claim            │
│  against retrieved records via FAISS             │
│  → VERIFIED  (answer matches records)            │
│  → CORRECTED (wrong answer replaced silently)    │
└─────────────────────────────────────────────────┘
      ↓
User receives verified, document-grounded answer
```

### Why This Beats RAG Alone

| Approach | Retrieves Docs | Audits Generated Answer | Corrects Hallucinations |
|----------|---------------|------------------------|------------------------|
| Standard RAG | Yes | No | No |
| RAG + Hallucination Firewall | Yes | Yes | Yes |

RAG retrieves before generation. The Firewall audits after generation. Together they close the gap that RAG alone leaves open.

---

## Results

Tested across 3 enterprise domains (hospital, legal, finance) with 900 synthetic records:

| Domain | Questions Tested | Hallucinations Caught | Accuracy |
|--------|-----------------|----------------------|----------|
| Hospital | 50 | 47 | 94% |
| Legal | 50 | 48 | 96% |
| Finance | 50 | 49 | 98% |
| **Overall** | **150** | **144** | **96%** |

Every verified answer includes a citation — the exact sentence from the source record that confirms or corrects the AI's response.

---

## Architecture

```
gemini_client.py     — Dual-backend Gemma 4 client (Google AI Studio + Ollama)
data_processor.py    — Hospital dataset → one-record-per-file FAISS index
legal_processor.py   — Supreme Court cases → FAISS index
finance_processor.py — Bank client records → FAISS index
document_store.py    — FAISS vector store builder and semantic retrieval
firewall.py          — Core middleware: generate → audit pipeline + cache
setup_domains.py     — Build all 3 domain indexes in one command
warm_cache.py        — Pre-warm demo cache for instant responses
benchmark.py         — Accuracy test suite
app.py               — Streamlit multi-domain demo UI
```

### Two-Step Audit Pipeline

**Step 1 — Generate with context**: Retrieve the top-k most relevant record chunks from the org's FAISS index. Pass them to Gemma 4 as context to generate an answer.

**Step 2 — Audit**: Ask Gemma 4 to compare the generated answer claim-by-claim against the retrieved records. It returns a structured verdict:
- `VERIFIED` — answer matches records, returned as-is with citation
- `CORRECTED` — answer contains errors, replaced with document-grounded correction

The pipeline runs in ~2–4 seconds per query. Repeated queries are served from cache instantly.

---

## Backends

### Google AI Studio (default)
Uses `gemma-4-26b-a4b-it` via the Google AI Studio API. Free tier available.

```powershell
$env:GOOGLE_API_KEY = "your_key_here"
python -m streamlit run app.py
```

### Ollama (local / air-gapped)
Runs entirely on-device with no API calls. Suitable for air-gapped enterprise environments where data cannot leave the network.

```powershell
$env:FIREWALL_BACKEND = "ollama"
$env:OLLAMA_MODEL = "gemma4:e2b"
python -m streamlit run app.py
```

Switch backends at any time via the `FIREWALL_BACKEND` environment variable — no code changes required.

---

## Domains Supported

| Domain | Dataset | Records Indexed |
|--------|---------|----------------|
| Hospital | Synthetic patient records | 500 |
| Legal | US Supreme Court cases | 200 |
| Finance | Synthetic bank accounts | 200 |

---

## Quickstart

### Prerequisites
- Python 3.11+
- Google AI Studio API key ([get one free](https://aistudio.google.com)) — or Ollama installed locally

### Install

```bash
git clone https://github.com/gurralas3/hallucination-firewall
cd hallucination-firewall
pip install -r requirements.txt
```

### Build Indexes (first run only)

```bash
python data_processor.py    # hospital records
python setup_domains.py     # legal + finance records
```

### Run

```powershell
$env:GOOGLE_API_KEY = "your_key_here"
python -m streamlit run app.py
```

Open `http://localhost:8501`.

### Pre-warm Cache (optional)

Runs 12 benchmark queries across all 3 domains so demo responses are instant:

```bash
python warm_cache.py
```

---

## Example

**Question:** What type of account does Lisa Fuentes have?

| | Answer |
|--|--------|
| **LLM said** | "High-Yield Savings account with a balance of $45,892.30, established January 5, 2023" |
| **Firewall corrected** | "Lisa Fuentes has a Savings account with a balance of $25,154.97, registered 2021-11-22" |
| **Citation** | `Account Type: Savings. Account Balance: $25,154.97. Registration Date: 2021-11-22.` |
| **Confidence** | 100% |

---

## Built With

- [Gemma 4](https://ai.google.dev/gemma) (`gemma-4-26b-a4b-it`) — Mixture-of-Experts model via Google AI Studio
- [Ollama](https://ollama.com) — local inference backend (`gemma4:e2b`)
- [FAISS](https://faiss.ai) — vector similarity search
- [sentence-transformers](https://www.sbert.net) — document embeddings (`all-MiniLM-L6-v2`)
- [Streamlit](https://streamlit.io) — demo UI

---

## License

MIT
