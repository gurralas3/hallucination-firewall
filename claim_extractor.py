from gemini_client import generate
import json

SYSTEM_PROMPT = """You are a claim extraction assistant.
Given an AI-generated response, extract the most important factual claims (maximum 3).
Return ONLY a valid JSON array of strings. No explanation, no markdown, no extra text.
Example output: ["Claim 1", "Claim 2", "Claim 3"]"""

def extract_claims(ai_answer: str) -> list[str]:
    raw = generate(
        contents=f"{SYSTEM_PROMPT}\n\nExtract all factual claims from this response:\n\n{ai_answer}",
        temperature=0.1
    )
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


if __name__ == "__main__":
    test_cases = [
        {
            "label": "E-commerce",
            "answer": "We offer free 30-day returns on all products. Our standard shipping takes 2 days and is completely free for all orders."
        },
        {
            "label": "Hospital",
            "answer": "Metformin is safe for patients with severe renal failure. Standard dose is 2000mg daily."
        }
    ]

    for case in test_cases:
        print(f"\n--- {case['label']} ---")
        claims = extract_claims(case["answer"])
        for i, claim in enumerate(claims, 1):
            print(f"  {i}. {claim}")
