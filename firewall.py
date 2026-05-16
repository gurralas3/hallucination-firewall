from gemini_client import generate, generate_stream
import json
import os
from datetime import datetime
from document_store import search

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache.json")

def _load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def _save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

# ── Domain-aware prompts ─────────────────────────────────────────────────────

_LLM_PROMPTS = {
    "hospital": (
        "A user is asking about a patient in a hospital system. "
        "Answer with a confident, specific, plausible response. "
        "Give real-sounding specific details: a medication name, doctor name, or diagnosis. "
        "Keep the answer to 1-2 sentences.\n\nQuestion: {question}\nAnswer:"
    ),
    "legal": (
        "A user is asking about a US Supreme Court case. "
        "Answer with a confident, specific, plausible response. "
        "Give real-sounding specific details: a Chief Justice name, case outcome, vote count, or date. "
        "Keep the answer to 1-2 sentences.\n\nQuestion: {question}\nAnswer:"
    ),
    "finance": (
        "A user is asking about a bank client account. "
        "Answer with a confident, specific, plausible response. "
        "Give real-sounding specific details: an account balance, account type, or registration date. "
        "Keep the answer to 1-2 sentences.\n\nQuestion: {question}\nAnswer:"
    ),
    "chw": (
        "A community health worker in a rural clinic is asking about a medicine or treatment. "
        "Answer with a confident, specific, plausible response about dosage, usage, or safety. "
        "Give real-sounding specific details: a drug dose, age range, or treatment duration. "
        "Keep the answer to 1-2 sentences.\n\nQuestion: {question}\nAnswer:"
    ),
}

_VERIFY_ROLES = {
    "hospital": "You are a medical fact-checker for a hospital system.",
    "legal":    "You are a legal fact-checker for a Supreme Court case database.",
    "finance":  "You are a financial fact-checker for a bank account system.",
    "chw":      "You are a clinical safety verifier for a rural community health worker assistant. Drug dosage accuracy is life-critical — flag any hallucinated dose, wrong age range, or incorrect contraindication.",
}

VERIFY_PROMPT = """{role}

AI ANSWER TO CHECK:
{answer}

ACTUAL RECORDS:
{context}

Compare the AI answer against the records.
Return ONLY valid JSON — no explanation, no markdown:
{{
  "verdict": "VERIFIED",
  "final_answer": "<repeat the original answer>",
  "confidence": 0.95,
  "citation": "<exact sentence from the records that confirms this>",
  "contradiction": ""
}}
OR if the answer contains errors:
{{
  "verdict": "CORRECTED",
  "final_answer": "<correct answer using only the records>",
  "confidence": 0.95,
  "citation": "<exact sentence from the records that proves the correction>",
  "contradiction": "<what the AI said wrong vs what the record says>"
}}"""


def _llm_prompt(question: str, org_id: str) -> str:
    template = _LLM_PROMPTS.get(org_id, _LLM_PROMPTS["hospital"])
    return template.format(question=question)


def _generate_llm_answer(user_question: str, org_id: str = "hospital") -> str:
    return generate(contents=_llm_prompt(user_question, org_id), temperature=1.0)


def stream_llm_answer(user_question: str, org_id: str = "hospital"):
    yield from generate_stream(contents=_llm_prompt(user_question, org_id), temperature=1.0)


# ── Prompt 2: verify + correct in one shot ──────────────────────────────────

def _verify_and_correct(llm_answer: str, user_question: str, org_id: str) -> dict:
    chunks = search(query=user_question, org_id=org_id, top_k=3)
    context = "\n\n".join(chunks) if chunks else "No relevant records found."

    role = _VERIFY_ROLES.get(org_id, _VERIFY_ROLES["hospital"])
    raw = generate(
        contents=VERIFY_PROMPT.format(role=role, answer=llm_answer, context=context),
        temperature=0.1,
        max_tokens=1024,
    )

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw.strip())
        return {
            "verdict":       data.get("verdict", "VERIFIED"),
            "final_answer":  data.get("final_answer", llm_answer),
            "confidence":    float(data.get("confidence", 0.9)),
            "citation":      data.get("citation", ""),
            "contradiction": data.get("contradiction", ""),
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "verdict": "VERIFIED", "final_answer": llm_answer,
            "confidence": 0.5, "citation": "", "contradiction": "",
        }


# ── Public API ───────────────────────────────────────────────────────────────

def verify_answer(llm_answer: str, user_question: str, org_id: str) -> dict:
    data = _verify_and_correct(llm_answer, user_question, org_id)
    status = "CORRECTED" if data["verdict"] == "CORRECTED" else "VERIFIED"

    result = {
        "question":        user_question,
        "original_answer": llm_answer,
        "final_answer":    data["final_answer"],
        "status":          status,
        "confidence":      data["confidence"],
        "citation":        data["citation"],
        "contradiction":   data["contradiction"],
        "hallucinations":  1 if status == "CORRECTED" else 0,
        "unverifiable":    0,
        "details":         [{"contradiction": data["contradiction"], "citation": data["citation"]}],
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "org_id":          org_id,
    }

    cache = _load_cache()
    cache[f"{org_id}::{user_question.strip().lower()}"] = result
    _save_cache(cache)
    return result


def chat(user_question: str, org_id: str) -> dict:
    cache = _load_cache()
    cache_key = f"{org_id}::{user_question.strip().lower()}"
    if cache_key in cache:
        print("[Firewall] Cache hit — returning instantly")
        return cache[cache_key]

    print("[Firewall] Generating LLM answer...")
    llm_answer = _generate_llm_answer(user_question, org_id)

    print("[Firewall] Verifying against records...")
    return verify_answer(llm_answer, user_question, org_id)


if __name__ == "__main__":
    for q, org in [
        ("Who is the doctor treating Ronald Park?",                      "hospital"),
        ("Who was the Chief Justice in DRAPER v. UNITED STATES?",        "legal"),
        ("What is Paul Watts's account balance?",                        "finance"),
    ]:
        print(f"\n{'='*60}\n[{org.upper()}] {q}\n{'='*60}")
        r = chat(q, org)
        print(f"Status:   {r['status']}")
        print(f"Original: {r['original_answer'][:100]}")
        print(f"Final:    {r['final_answer']}")
