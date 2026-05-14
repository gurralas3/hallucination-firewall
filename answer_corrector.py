from gemini_client import generate
from document_store import search

SAFE_FALLBACK = "I don't have verified information on this. Please contact our support team directly."

SYSTEM_PROMPT = """You are a helpful hospital assistant.
Answer the question using ONLY the information provided in the patient records below.
Do not add any information not present in the records.
Be concise, clear, and professional."""

def correct_answer(user_question: str, org_id: str) -> str:
    chunks = search(query=user_question, org_id=org_id, top_k=3)

    if not chunks:
        return SAFE_FALLBACK

    context = "\n\n".join(chunks)

    return generate(
        contents=f"""{SYSTEM_PROMPT}

Patient Records:
{context}

Question: {user_question}

Answer based only on the patient records above:""",
        temperature=0.1
    )


if __name__ == "__main__":
    print("--- Test: Question covered in records ---")
    answer = correct_answer("Who is the doctor treating Ronald Park?", "hospital")
    print(f"Corrected Answer: {answer}\n")
