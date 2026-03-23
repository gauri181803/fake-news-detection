"""
test_api.py — Smoke-tests for the Fake News Detection API.

Run after starting the server:
    uvicorn app.main:app --port 8000
    python test_api.py
"""

import requests
import json

BASE = "http://localhost:8000"

# ── Colour helpers ─────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"


def ok(msg):   print(f"  {GREEN}✅ PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}❌ FAIL{RESET}  {msg}")
def info(msg): print(f"  {YELLOW}ℹ{RESET}  {msg}")


# ── Test cases ─────────────────────────────────────────────────
TESTS = [
    # (description, text, language, expected_label_hint)
    (
        "Hindi — Real news (political)",
        "नरेंद्र मोदी ने आज नई दिल्ली में एक महत्वपूर्ण बैठक की अध्यक्षता की। "
        "केंद्र सरकार ने इस बैठक में कई नीतिगत फैसले लिए।",
        "hi",
        None,   # just check API responds
    ),
    (
        "Hindi — Fake (sensationalist)",
        "सनसनीखेज खुलासा! भारत सरकार ने सभी बैंक खाते बंद करने का आदेश दिया, "
        "कल से नहीं निकाल पाएंगे पैसे। वायरल सच्चाई जानें।",
        "hi",
        None,
    ),
    (
        "Marathi — Real news",
        "पुणे येथे महाराष्ट्र सरकारने नवीन पायाभूत सुविधा प्रकल्पाचे उद्घाटन केले. "
        "मुख्यमंत्र्यांनी नागपूर येथे आढावा बैठक घेतली.",
        "mr",
        None,
    ),
    (
        "Gujarati — Real news",
        "ગુજરાત સરકારે અમદાવાદમાં નવી શૈક્ષણિક નીતિ જાહેર કરી. "
        "મુખ્યમંત્રીએ સુરતમાં આ અંગે જાહેરાત કરી.",
        "gu",
        None,
    ),
    (
        "Telugu — Real news",
        "తెలంగాణ ప్రభుత్వం హైదరాబాద్‌లో కొత్త పారిశ్రామిక విధానాన్ని ప్రకటించింది. "
        "ముఖ్యమంత్రి విజయవాడలో సమావేశాన్ని నిర్వహించారు.",
        "te",
        None,
    ),
    (
        "Auto language detection (Hindi, no lang param)",
        "भारतीय अंतरिक्ष अनुसंधान संगठन ने चंद्रयान मिशन की सफलता की घोषणा की।",
        None,   # omit language — test auto-detect
        None,
    ),
    (
        "Short text (should return 422)",
        "hello",
        "hi",
        "error",
    ),
]


def test_health():
    print("\n── /health ──────────────────────────────────────────────")
    try:
        r = requests.get(f"{BASE}/health", timeout=10)
        if r.status_code == 200:
            data = r.json()
            ok(f"Status: {data['status']}  |  Models loaded: {data['models_loaded']}")
        else:
            fail(f"HTTP {r.status_code}")
    except Exception as e:
        fail(f"Could not connect: {e}")


def test_predict(description, text, language, expected):
    print(f"\n  [{description}]")
    payload = {"text": text}
    if language:
        payload["language"] = language

    try:
        r = requests.post(f"{BASE}/predict", json=payload, timeout=30)

        if expected == "error":
            if r.status_code in (400, 422):
                ok(f"Correctly returned HTTP {r.status_code}")
            else:
                fail(f"Expected error but got HTTP {r.status_code}")
            return

        if r.status_code != 200:
            fail(f"HTTP {r.status_code}: {r.text[:200]}")
            return

        data = r.json()
        label      = data["prediction"]
        conf       = data["confidence"]
        lang_det   = data["language_detected"]
        entities   = data["entities_found"]
        v_count    = data["verified_count"]
        explanation= data["explanation"]

        ok(
            f"Prediction: {label} ({conf:.1%})  |  "
            f"Lang: {lang_det}  |  "
            f"Entities: {len(entities)} ({v_count} verified)"
        )
        info(f"  {explanation}")

        if expected and expected != label:
            fail(f"Expected {expected} but got {label}")

    except Exception as e:
        fail(f"Exception: {e}")


def run_all():
    print("=" * 60)
    print("  Multilingual Fake News API — Test Suite")
    print("=" * 60)

    test_health()

    print("\n── /predict ─────────────────────────────────────────────")
    for desc, text, lang, expected in TESTS:
        test_predict(desc, text, lang, expected)

    print("\n" + "=" * 60)
    print("  Test run complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
