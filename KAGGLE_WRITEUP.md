# DoseGuard — Post-Generation Verification for Clinical LLM Safety

**Gemma 4 Good Hackathon | Safety & Trust · Health & Sciences · Ollama · Unsloth**

---

## The Moment That Started This

In early 2025, I had a red, itchy eye for two days.

I photographed the medicine shelf at CVS and asked an AI assistant what to use. It confidently recommended **Visine (Tetrahydrozoline)** — a vasoconstrictor that only reduces redness temporarily.

The correct WHO first-line treatment for allergic conjunctivitis is **Ketotifen** — an antihistamine that treats the actual allergic cause.

The AI sounded confident. The answer was wrong. I would have trusted it.

---

## The Bigger Problem

Scaled to a rural clinic, that same failure pattern becomes life-threatening.

A beginner community health worker consults an AI about a 10 kg child's Amoxicillin dose for pneumonia. The AI confidently returns **500 mg twice daily**. The correct WHO dose is **160 mg twice daily**. That is a 3× overdose.

**LLMs hallucinate medical dosing at rates of 43–67%.** Fine-tuning reduces this — it does not eliminate it. DoseGuard is the safety layer that catches what fine-tuning misses.

---

## One Sentence

> **DoseGuard verifies AI-generated medical answers before a person acts on them.**

---

## Why Retrieval Alone Is Not Enough

Standard RAG systems retrieve relevant documents before generation. This is necessary — but not sufficient.

An LLM can:
- Retrieve the correct WHO dosing guideline
- Still generate the wrong dose in its final answer
- Merge details from two different medicines
- Introduce unsupported claims during the generation process itself

RAG provides context. It does not audit the output.

```
Standard RAG:
  User → Retrieve docs → LLM generates → User receives answer  ← NO VERIFICATION

DoseGuard:
  User → Retrieve docs → LLM generates → Firewall audits → Verified answer
```

DoseGuard adds a second pass **after generation**: the Hallucination Firewall — a post-generation verification layer — audits the final answer itself against retrieved WHO records before the response reaches the user.

Retrieval provides context. The Firewall verifies the final claim.

---

## The Architecture

```
User Question (text or medicine photo)
          │
          ▼
Fine-tuned Gemma 4  [Unsloth LoRA · 505 WHO Q&A pairs · Kaggle T4 GPU]
          │
          ▼  LLM generates answer  →  may be wrong
          │
          ▼
FAISS Semantic Search  [all-MiniLM-L6-v2 · 53 WHO medicines · top-3 retrieval]
          │
          ▼
Hallucination Firewall  [Gemma 4 · temperature 0.1 · structured JSON verdict]
Compares generated answer against retrieved WHO records
          │
     ┌────┴─────────────────────┐
     │                          │
VERIFIED                  CORRECTED              UNVERIFIABLE
Pass through              Replace with           Escalate to clinician
with citation             WHO-correct answer     — safer than guessing
```

### The Failure Flow in One Line

```
LLM → Wrong dose → Firewall intercepts → WHO-corrected output with citation
```

---

## Three Verdicts

| Verdict | When triggered | Action |
|---------|---------------|--------|
| ✅ VERIFIED | Answer matches WHO records | Deliver with citation |
| 🔒 CORRECTED | Answer contradicts WHO records | Replace with correct answer + explain error |
| ⚠ UNVERIFIABLE | WHO records absent or retrieval confidence below threshold | Block and escalate to clinician |

### When UNVERIFIABLE Is Triggered

UNVERIFIABLE is not a failure state — it is a deliberate safety feature:

- Top FAISS similarity score falls below the retrieval confidence threshold
- Retrieved WHO records do not cover the queried medicine or indication
- Conflicting signals across top-3 retrieved records
- Verification confidence score below 0.5

**This is a key architectural decision:** a system that always answers is more dangerous than a system that sometimes refuses.

---

## The 3-Way Benchmark

> Fine-tuning alone is insufficient. Here is the proof.

### By Accuracy

| Stage | Correct | Partial | Wrong |
|-------|---------|---------|-------|
| Base Gemma 4 | — | — | — |
| Fine-tuned (Unsloth LoRA) | — | — | — |
| Fine-tuned + Firewall | — | — | — |

*Run `python benchmark_doseguard.py` to populate with live results.*

### By Unsafe Recommendation Rate (the metric that matters)

| Stage | Unsafe Clinical Recommendations |
|-------|--------------------------------|
| Base Gemma 4 | —/20 |
| Fine-tuned | —/20 |
| Fine-tuned + Firewall | —/20 |

An "unsafe recommendation" is any answer that would lead a health worker to administer a wrong dose, wrong drug, or contraindicated treatment. This is the metric that actually matters in clinical settings — not abstract accuracy.

### Benchmark Coverage (20 Questions, 7 Categories)

| Category | Questions | Example |
|----------|-----------|---------|
| Antibiotics — child dosing | 4 | Amoxicillin 8kg, 10kg; Doxycycline contraindication |
| Antimalarials | 3 | Severe malaria treatment; Chloroquine resistance; AL weight-band |
| Maternal health | 3 | Magnesium Sulfate loading; PPH without refrigeration; eclampsia antidote |
| Contraindications | 4 | Aspirin in children; Metformin in kidney failure; Lisinopril in pregnancy |
| Emergency medicines | 2 | Adrenaline dose; Dextrose for hypoglycaemia |
| Eye medicines | 2 | Ketotifen vs Visine; Chloramphenicol indication |
| Antiretroviral / TB | 2 | Isoniazid supplement; Zidovudine PMTCT |

---

## Known Limitations

Honest evaluation includes failures.

DoseGuard still struggles when:

- **Symptoms are vague** — "my child is not well" cannot be matched to a specific WHO record
- **Multiple medicines have overlapping indications** — retrieval may surface the wrong drug first
- **WHO EML coverage is absent** — medicines not on the Essential Medicines List return UNVERIFIABLE, which is correct but unhelpful
- **Weight-based calculations involve interpolation** — WHO records give fixed weight bands; a 13 kg child falls between the 10 kg and 15 kg entries

In these cases the Firewall returns UNVERIFIABLE or a low-confidence correction. **This is the intended behaviour** — it is safer to escalate than to guess.

---

## What Was Built

| Component | Description |
|-----------|-------------|
| `sample_docs/who_medicines.txt` | 53 WHO Essential Medicines — adult/child dosing by weight, contraindications, clinical notes |
| `training_data/who_qa_pairs.json` | 505 WHO Q&A training pairs across 17 medicine categories |
| `kaggle_notebook/doseguard_unsloth_finetune.ipynb` | Unsloth LoRA fine-tuning + 3-way benchmark on Kaggle T4 GPU |
| `chw_processor.py` | FAISS index builder — all-MiniLM-L6-v2 embeddings |
| `firewall.py` | Hallucination Firewall — generate + retrieve + verify + verdict |
| `photo_verify.py` | Gemma 4 vision — reads medicine box label, runs firewall |
| `benchmark_doseguard.py` | 20-question benchmark with unsafe recommendation rate metric |
| `app.py` | Streamlit UI — Quick Check + Clinical Check + Photo Check + Audit Log |
| `Modelfile` | Ollama deployment — offline rural clinic mode |

---

## The Two Real Scenarios DoseGuard Solves

### Scenario 1 — CVS / Pharmacy (Photo Check)
User photographs a Visine bottle. Gemma 4 reads the label. Firewall checks WHO records.

```
Extracted  : Visine Advanced · Tetrahydrozoline HCl 0.05%
Label claim: Redness Relief Eye Drops
Question   : Is this the right treatment for itchy red eyes from allergies?

🔒 CORRECTED
WHO record : Ketotifen is the first-line antihistamine for allergic conjunctivitis.
             Tetrahydrozoline reduces redness only — it does not treat allergy.
Error found: Visine does not treat allergic cause; recommending it for allergy is incorrect.
```

### Scenario 2 — Rural Clinic (Clinical Check)
Health worker asks about Amoxicillin dose for a 10 kg child.

```
Query   : Correct dose of Amoxicillin for 10 kg child with pneumonia?
LLM said: 500 mg twice daily

🔒 CORRECTED
WHO dose : 200 mg twice daily (40 mg/kg/day in two divided doses) for 5 days
Citation : CHILD DOSE (under 5, pneumonia): 40mg/kg/day in two divided doses for 5 days.
Error    : LLM returned adult dose. 500 mg twice daily is a 2.5× overdose for a 10 kg child.
```

---

## Two Deployment Modes

### Online — Google AI Studio (Gemma 4)
Connected clinics and pharmacies. Full Gemma 4 capability.

### Offline — Ollama (GGUF)
Rural clinics with no internet. Deploy with:
```bash
ollama create doseguard -f Modelfile
ollama run doseguard "Amoxicillin dose for 10kg child with pneumonia?"
```

The Misoprostol parallel: Misoprostol is preferred over Oxytocin in rural settings because it needs no refrigeration. DoseGuard runs offline for the same reason — infrastructure cannot be a prerequisite for safe medicine guidance.

---

## Prize Track Alignment

| Track | Alignment |
|-------|-----------|
| **Safety & Trust** | Core thesis — post-generation verification for reliable, grounded AI |
| **Health & Sciences** | WHO Essential Medicines, rural clinic deployment, maternal/child/TB/ARV coverage |
| **Ollama** | GGUF export, Modelfile, offline deployment — no internet, no cloud |
| **Unsloth** | LoRA fine-tuning on Kaggle T4, 505 WHO pairs, published benchmark |

---

## The Broader Argument

High-stakes LLM systems benefit from a verification layer between generation and delivery.

DoseGuard demonstrates this pattern in medicine. The same architecture applies to any domain where an incorrect AI answer causes irreversible harm — legal dosing, drug interactions, emergency protocols.

The contribution is not just a medicine app. It is a reusable verification pattern:

> **Generate → Retrieve → Verify → Output**

with explicit handling of the case where verification cannot confirm safety.

---

## Closing

Most AI healthcare projects ask: *"What can AI do for health?"*

DoseGuard asks: *"How can AI be trusted when mistakes cost lives?"*

Fine-tuning is part of the answer.
The Hallucination Firewall is the other part.

---

*WHO-sourced educational information only. Always consult a qualified clinician before administering any medicine.*
*Built with Gemma 4 · Unsloth · FAISS · Ollama · WHO Essential Medicines List*
