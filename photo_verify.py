"""
DoseGuard Photo Verification — Multimodal Medicine Box Analysis
Scenario: User photographs a medicine box label and asks if it's appropriate.

Uses Gemma 4's vision capabilities to:
1. Read the medicine name and active ingredients from the photo
2. Ask what the medicine is used for
3. Run through the Hallucination Firewall to verify against WHO records
4. Return VERIFIED / CORRECTED / UNVERIFIABLE

Demo scenario:
  User: "Is this eye drop the right treatment for my itchy red eye from allergies?"
  Photo: Tetrahydrozoline (Visine) bottle
  Result: CORRECTED — Visine treats redness only; Ketotifen is WHO first-line for allergic conjunctivitis
"""

import os
import base64
import json
from pathlib import Path
from firewall import verify_answer

# ── Google AI Studio vision (Gemma 4 multimodal) ─────────────────────────────

def _encode_image(image_path: str) -> tuple[str, str]:
    """Return (base64_data, mime_type)."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp"}
    mime = mime_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def read_medicine_from_image(image_path: str, user_question: str) -> dict:
    """
    Send image to Gemma 4 vision to extract medicine info.
    Returns dict with medicine_name, ingredients, claimed_use, extracted_text.
    """
    backend = os.environ.get("FIREWALL_BACKEND", "google").lower()

    extraction_prompt = f"""You are reading a medicine box or bottle label in a photograph.

The user's question is: {user_question}

From this image, extract:
1. The medicine name (brand and/or generic)
2. The active ingredient(s)
3. What the label says it treats or is used for
4. Any dosage information visible on the label

Return ONLY valid JSON:
{{
  "medicine_name": "<brand name>",
  "generic_name": "<active ingredient>",
  "label_claim": "<what the label says it treats>",
  "label_dose": "<dosage on label if visible>",
  "confidence": 0.95,
  "raw_text": "<any other relevant text from the label>"
}}

If you cannot read the label clearly, return confidence below 0.5."""

    if backend == "ollama":
        # Ollama vision — use llava or gemma with vision
        import ollama as _ollama
        ollama_model = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")

        try:
            response = _ollama.chat(
                model=ollama_model,
                messages=[{
                    "role": "user",
                    "content": extraction_prompt,
                    "images": [image_path],
                }],
            )
            raw = (response.message.content or "").strip()
        except Exception as e:
            return {"error": str(e), "medicine_name": "", "generic_name": "",
                    "label_claim": "", "confidence": 0.0}
    else:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        b64, mime = _encode_image(image_path)

        try:
            response = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                contents=[
                    types.Part.from_bytes(data=base64.b64decode(b64), mime_type=mime),
                    extraction_prompt,
                ],
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=512),
            )
            raw = (response.text or "").strip()
        except Exception as e:
            return {"error": str(e), "medicine_name": "", "generic_name": "",
                    "label_claim": "", "confidence": 0.0}

    # Parse JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        return {"medicine_name": raw[:100], "generic_name": "",
                "label_claim": "", "confidence": 0.3, "raw_text": raw}


def verify_photo(image_path: str, user_question: str) -> dict:
    """
    Full pipeline: photo → medicine extraction → WHO firewall verification.

    Returns:
        {
          "extracted": {...}       # what was read from the photo
          "firewall": {...}        # VERIFIED / CORRECTED / UNVERIFIABLE
          "full_question": str     # the constructed query
        }
    """
    print(f"[PhotoVerify] Reading medicine label from: {image_path}")
    extracted = read_medicine_from_image(image_path, user_question)

    if extracted.get("confidence", 0) < 0.3:
        return {
            "extracted": extracted,
            "firewall": {
                "status": "UNVERIFIABLE",
                "final_answer": "Could not read the medicine label clearly. Please try a clearer photo.",
                "confidence": 0.1,
                "citation": "",
                "contradiction": "",
            },
            "full_question": user_question,
        }

    med_name = extracted.get("medicine_name") or extracted.get("generic_name", "unknown medicine")
    label_claim = extracted.get("label_claim", "")

    # Construct verification query from what was read
    full_question = f"{user_question} The medicine label shows: {med_name}"
    if extracted.get("generic_name"):
        full_question += f" (active ingredient: {extracted['generic_name']})"
    if label_claim:
        full_question += f". The label claims it is used for: {label_claim}."

    print(f"[PhotoVerify] Extracted: {med_name} | Claim: {label_claim}")
    print(f"[PhotoVerify] Verification query: {full_question[:120]}...")

    firewall_result = verify_answer(
        llm_answer=f"{med_name} is used for {label_claim}. {extracted.get('raw_text', '')}",
        user_question=full_question,
        org_id="chw",
    )

    return {
        "extracted": extracted,
        "firewall":  firewall_result,
        "full_question": full_question,
    }


# ── Demo scenario ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python photo_verify.py <image_path> [question]")
        print()
        print("Demo: Simulating CVS scenario without actual image...")

        # Simulate what would happen with a Visine bottle photo
        # (for demo/testing without actual image file)
        demo_extracted = {
            "medicine_name": "Visine Advanced Redness + Irritation Relief",
            "generic_name": "Tetrahydrozoline HCl 0.05%",
            "label_claim": "Redness Relief Eye Drops — for red, irritated eyes",
            "label_dose": "1-2 drops up to 4 times daily",
            "confidence": 0.95,
        }

        user_q = "Is this eye drop the right treatment for itchy red eyes from allergies?"
        med = demo_extracted["medicine_name"]
        label = demo_extracted["label_claim"]
        full_q = f"{user_q} The medicine label shows: {med} (active ingredient: {demo_extracted['generic_name']}). The label claims: {label}."

        print(f"\nSimulated photo: {med}")
        print(f"User question  : {user_q}")
        print(f"Firewall query : {full_q[:100]}...\n")

        result = verify_answer(
            llm_answer=f"{med} contains {demo_extracted['generic_name']} and is indicated for {label}.",
            user_question=full_q,
            org_id="chw",
        )

        print(f"Status     : {result['status']}")
        print(f"Final answer: {result['final_answer']}")
        print(f"Contradiction: {result.get('contradiction', '')}")
        print(f"WHO citation : {result.get('citation', '')}")

    else:
        image_path = sys.argv[1]
        question   = sys.argv[2] if len(sys.argv) > 2 else "Is this medicine appropriate for my condition?"
        result = verify_photo(image_path, question)
        print(json.dumps(result, indent=2))
