# Hallucination Firewall — Post-Generation Verification for Enterprise AI

**Gemma 4 Good Hackathon | Safety & Trust**

---

## The Problem

Enterprise AI deployments across hospitals, legal systems, and financial institutions share one critical vulnerability: LLMs generate confident, plausible-sounding answers that are factually wrong.

A hospital system asks who is treating a patient. The AI invents a doctor's name.
A legal team asks about a Supreme Court case outcome. The AI fabricates a vote count.
A bank asks about a client's account balance. The AI returns a convincing but wrong figure.

Most safeguards focus on what goes *into* the model — retrieval, prompt conditioning, grounding. Far fewer systems independently audit the final answer before it reaches the user.

> **The Hallucination Firewall audits every AI-generated answer before delivery.**

---

## One Sentence

> A post-generation verification layer that intercepts LLM answers, checks them against authoritative records, and either confirms, corrects, or blocks them before the user sees the response.

---

## Why RAG Alone Is Not Enough

```
Standard RAG:
  User → Retrieve docs → LLM generates → User receives answer  ← NO VERIFICATION

Hallucination Firewall:
  User → Retrieve docs → LLM generates → Firewall audits → Verified answer
```

An LLM can retrieve the correct record and still generate the wrong answer. It can merge details from two records. It can introduce unsupported claims mid-sentence. RAG provides context. The Firewall verifies the final claim.

---

## Why Verification Instead of Just Better Prompting?

Prompting influences generation probabilistically. Verification evaluates the final claim deterministically against retrieved evidence.

The generator optimizes for producing plausible language. The verifier optimizes for contradiction detection against retrieved records. **Separating these objectives reduces the likelihood that fluent hallucinations pass unchecked.**

The Hallucination Firewall treats every LLM output as untrusted until it passes independent evidence review.

---

## The Architecture

```
User Question
      │
      ▼
Gemma 4 generates answer  [temperature 1.0 — confident, may hallucinate]
      │
      ▼
FAISS Semantic Search  [all-MiniLM-L6-v2 · top-3 nearest records from org store]
      │
      ▼
Hallucination Firewall  [Gemma 4 · temperature 0.1 · structured JSON verdict]
Compares generated answer against retrieved records
      │
 ┌────┴──────────────────────┐
 │                           │
VERIFIED               CORRECTED            UNVERIFIABLE
Pass through           Replace with         Block — escalate
with citation          record-grounded      to human review
                       answer
```

**Generate → Retrieve → Verify**

---

## Three Verdicts

| Verdict | When triggered | Action |
|---------|---------------|--------|
| ✅ VERIFIED | Answer matches retrieved records | Deliver with citation |
| 🔒 CORRECTED | Answer contradicts retrieved records | Replace with grounded answer |
| ⚠ UNVERIFIABLE | Retrieval similarity below threshold | Block and escalate |

> **A system that always answers is more dangerous than one that sometimes refuses.**

UNVERIFIABLE is a deliberate safety feature — when FAISS similarity falls below threshold (cosine similarity < 0.5), the Firewall refuses to guess. Escalation to a human reviewer is the correct output.

---

## Three Live Domains

| Domain | Example question | Hallucination caught |
|--------|-----------------|----------------------|
| **Hospital** | "Who is treating Ronald Park?" | Fabricated doctor name |
| **Legal** | "Who was Chief Justice in Draper v. United States?" | Invented case outcome |
| **Finance** | "What is Paul Watts's account balance?" | Made-up account details |

Each domain has its own FAISS index, its own document store, and its own verifier role — the Firewall adapts its verification lens to the domain it is checking.

---

## Live Examples

### Hospital
```
Question : Who is the doctor treating Ronald Park?
LLM said : Dr. Emily Chen is treating Ronald Park.

🔒 CORRECTED
Record   : Ronald Park is assigned to Dr. James Osei.
Error    : LLM fabricated a doctor name not present in any hospital record.
```

### Legal
```
Question : Who was the Chief Justice in Draper v. United States?
LLM said : Chief Justice Warren Burger presided, 7-2 decision.

🔒 CORRECTED
Record   : Chief Justice Earl Warren. Decision was unanimous.
Error    : Wrong Chief Justice and wrong vote count.
```

### Finance
```
Question : What is Paul Watts's account balance?
LLM said : Paul Watts has a balance of $24,310.

🔒 CORRECTED
Record   : Paul Watts — Savings Account — $8,750.
Error    : LLM invented a balance not in any account record.
```

---

## What Was Built

| Component | Description |
|-----------|-------------|
| `document_store.py` | FAISS index builder — all-MiniLM-L6-v2 · per-org isolation |
| `firewall.py` | Hallucination Firewall — generate + retrieve + verify + verdict |
| `gemini_client.py` | Gemma 4 client — generate + streaming |
| `app.py` | Streamlit UI — domain selector · evidence expander · audit log |
| `setup_domains.py` | One-command domain setup — indexes all org documents |
| `sample_docs/` | Hospital, legal, and finance document stores |

---

## The Retrieval Confidence Layer

Every response surfaces the retrieved records with similarity scores. When the Firewall returns UNVERIFIABLE, the UI shows exactly why — the best matching record was only X% similar, below the verification threshold.

This gives operators full transparency: not just a verdict, but the evidence behind it.

---

## What the Firewall Is (and Is Not)

| The Firewall is | The Firewall is not |
|---|---|
| A verification layer that reduces hallucinated outputs | A guarantee of correctness |
| A tool that surfaces uncertainty explicitly | A replacement for human review |
| Domain-agnostic — adapts to any org document store | A solution to all LLM failures |
| A post-generation audit before user delivery | A prompt engineering technique |

---

## The Broader Argument

Every organisation deploying LLMs needs a verification layer between generation and delivery. The Hallucination Firewall demonstrates this pattern across three domains. The same architecture applies to any domain where an incorrect AI answer causes irreversible harm — clinical dosing, legal rulings, financial records, emergency protocols.

The contribution is not just an application. It is a reusable verification pattern:

> **Generate → Retrieve → Verify**

with explicit handling of the case where verification cannot confirm safety.

---

## Closing

Most AI safety work asks: *"How do we make the model generate better answers?"*

The Hallucination Firewall asks: *"How do we catch the wrong answers before they reach anyone?"*

> **The question is no longer whether AI can answer. The question is whether you can trust the answer it gives.**

---

*Built with Gemma 4 · FAISS · Sentence Transformers · Streamlit*
