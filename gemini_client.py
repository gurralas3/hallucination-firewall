import os
import time

BACKEND = os.environ.get("FIREWALL_BACKEND", "google").lower()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")

if BACKEND == "ollama":
    import ollama as _ollama
else:
    from google import genai
    from google.genai import types
    _google_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    GOOGLE_MODEL = "gemma-4-26b-a4b-it"


def generate_stream(contents: str, temperature: float = 0.9):
    if BACKEND == "ollama":
        temperature = min(temperature, 0.7)  # thinking model returns empty content above ~0.8
        for chunk in _ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": contents}],
            stream=True,
            options={"temperature": temperature},
        ):
            text = chunk.message.content
            if text:
                yield text
    else:
        for chunk in _google_client.models.generate_content_stream(
            model=GOOGLE_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(temperature=temperature),
        ):
            if chunk.text:
                yield chunk.text


def generate(contents: str, temperature: float = 0.1, retries: int = 5, max_tokens: int = 512) -> str:
    if BACKEND == "ollama":
        temperature = min(temperature, 0.7)  # thinking model returns empty content above ~0.8
        for attempt in range(retries):
            try:
                response = _ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "user", "content": contents}],
                    options={"temperature": temperature, "num_predict": max_tokens},
                )
                return (response.message.content or "").strip()
            except Exception:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
    else:
        for attempt in range(retries):
            try:
                response = _google_client.models.generate_content(
                    model=GOOGLE_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                return (response.text or "").strip()
            except Exception as e:
                err = str(e)
                is_transient = any(code in err for code in ("500", "502", "503", "504"))
                if is_transient and attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise


def active_backend() -> str:
    if BACKEND == "ollama":
        return f"Ollama · {OLLAMA_MODEL} · local"
    return "Google AI Studio · gemma-4-26b-a4b-it"
