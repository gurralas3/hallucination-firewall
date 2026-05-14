import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from claim_extractor import extract_claims
from document_store import build_index, search
from answer_corrector import correct_answer, SAFE_FALLBACK

# ── Claim Extractor Tests ──────────────────────────────────────────────────────

def test_extract_claims_returns_list():
    answer = "We offer free returns. Shipping is free."
    claims = extract_claims(answer)
    assert isinstance(claims, list)
    assert len(claims) > 0

def test_extract_claims_not_empty_string():
    answer = "All sales are final and we do not offer refunds."
    claims = extract_claims(answer)
    for claim in claims:
        assert isinstance(claim, str)
        assert len(claim.strip()) > 0

def test_extract_claims_max_three():
    answer = "We offer 30-day returns. Shipping is free. We ship worldwide. Support is 24/7. No hidden fees."
    claims = extract_claims(answer)
    assert len(claims) <= 3

# ── Document Store Tests ───────────────────────────────────────────────────────

def test_build_and_search_index(tmp_path):
    doc = tmp_path / "policy.txt"
    doc.write_text("Refund Policy: All sales are final. No refunds offered.")

    org_id = "test_org_pytest"
    build_index(org_id, [str(doc)])

    results = search("refund policy", org_id)
    assert isinstance(results, list)
    assert len(results) > 0
    assert any("refund" in r.lower() or "sales" in r.lower() for r in results)

def test_search_returns_relevant_chunks(tmp_path):
    doc = tmp_path / "shipping.txt"
    doc.write_text("Shipping Policy: Standard shipping takes 5 to 7 business days.")

    org_id = "test_org_shipping"
    build_index(org_id, [str(doc)])

    results = search("how long does shipping take", org_id)
    assert len(results) > 0

# ── Answer Corrector Tests ─────────────────────────────────────────────────────

def test_corrector_returns_string():
    answer = correct_answer("What is your refund policy?", "techcorp")
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0

def test_corrector_uses_document_content():
    answer = correct_answer("What is your refund policy?", "techcorp")
    answer_lower = answer.lower()
    # Should mention something about sales or refunds from the actual document
    assert any(word in answer_lower for word in ["final", "refund", "sales", "exchange"])

def test_corrector_fallback_for_unknown_topic():
    answer = correct_answer("What is the meaning of life?", "techcorp")
    # Either safe fallback or an honest "I don't have that information"
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0

# ── Integration Test ───────────────────────────────────────────────────────────

def test_full_pipeline_claim_to_correction():
    ai_answer = "We offer free 30-day returns on all products."
    claims = extract_claims(ai_answer)
    assert len(claims) > 0

    results = search(claims[0], "techcorp")
    assert isinstance(results, list)

    corrected = correct_answer("What is your refund policy?", "techcorp")
    assert isinstance(corrected, str)
