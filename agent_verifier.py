from gemini_client import generate
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from document_store import search

CONFIDENCE_THRESHOLD = 0.70

VERIFY_SYSTEM = """You are a medical records verification agent.
Given a claim and relevant patient records, determine if the claim is accurate.

Respond with ONLY a valid JSON object in this exact format:
{
  "verdict": "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE",
  "confidence": 0.0-1.0,
  "citation": "exact quote from documents that supports this verdict",
  "risk": "HIGH" | "MEDIUM" | "LOW",
  "reason": "brief explanation"
}

Rules:
- SUPPORTED: documents clearly confirm the claim
- CONTRADICTED: documents clearly contradict the claim
- UNVERIFIABLE: claim cannot be confirmed or denied from documents
- HIGH risk if CONTRADICTED, MEDIUM if UNVERIFIABLE, LOW if SUPPORTED"""

def _run_agent_for_claim(claim: str, org_id: str) -> dict:
    chunks = search(query=claim, org_id=org_id, top_k=3)
    context = "\n\n".join(chunks) if chunks else "No relevant documents found."

    prompt = f"""{VERIFY_SYSTEM}

CLAIM TO VERIFY: {claim}

RELEVANT PATIENT RECORDS:
{context}

Verify the claim against the patient records above and return the JSON verdict."""

    raw = generate(contents=prompt, temperature=0.1)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
        confidence = float(data.get("confidence", 0.5))
        return {
            "claim":          claim,
            "verdict":        data.get("verdict", "UNVERIFIABLE"),
            "confidence":     confidence,
            "citation":       data.get("citation", ""),
            "risk":           data.get("risk", "MEDIUM"),
            "reason":         data.get("reason", ""),
            "low_confidence": confidence < CONFIDENCE_THRESHOLD
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "claim":          claim,
            "verdict":        "UNVERIFIABLE",
            "confidence":     0.0,
            "citation":       "",
            "risk":           "MEDIUM",
            "reason":         "Could not parse verification response.",
            "low_confidence": True
        }

def verify_all_claims(claims: list[str], org_id: str) -> list[dict]:
    results = [None] * len(claims)
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_index = {
            executor.submit(_run_agent_for_claim, claim, org_id): i
            for i, claim in enumerate(claims)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            results[idx] = future.result()
    return results
