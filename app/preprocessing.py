"""
Multilingual preprocessing pipeline.
Mirrors the exact preprocessing used in the training notebooks.
"""

import re
import time
import requests
from typing import List, Dict, Tuple

# ── Language metadata ──────────────────────────────────────────
LANG_META = {
    "hi": {"name": "Hindi",    "wikidata_code": "hi"},
    "mr": {"name": "Marathi",  "wikidata_code": "mr"},
    "gu": {"name": "Gujarati", "wikidata_code": "gu"},
    "te": {"name": "Telugu",   "wikidata_code": "te"},
}

# ── Named-entity regex patterns per language ───────────────────
# These mirror the regex NER used in the notebooks
NER_PATTERNS: Dict[str, Dict[str, str]] = {
    "hi": {
        "PERSON":   r"[A-Z][a-z]+ [A-Z][a-z]+",
        "ORG":      r"(?:सरकार|मंत्रालय|विभाग|संस्था|पार्टी|कंपनी|संगठन|बोर्ड|आयोग)",
        "LOCATION": r"(?:भारत|दिल्ली|मुंबई|कोलकाता|चेन्नई|बेंगळुरू|हैदराबाद|पुणे|जयपुर|लखनऊ|पाकिस्तान|चीन|अमेरिका)",
        "DATE":     r"(?:जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर|\d{4})",
    },
    "mr": {
        "PERSON":   r"[A-Z][a-z]+ [A-Z][a-z]+",
        "ORG":      r"(?:सरकार|मंत्रालय|संस्था|पक्ष|कंपनी|संघटना|मंडळ|आयोग)",
        "LOCATION": r"(?:भारत|मुंबई|पुणे|नागपूर|नाशिक|दिल्ली|कोल्हापूर|सोलापूर|औरंगाबाद|ठाणे)",
        "DATE":     r"(?:जानेवारी|फेब्रुवारी|मार्च|एप्रिल|मे|जून|जुलै|ऑगस्ट|सप्टेंबर|ऑक्टोबर|नोव्हेंबर|डिसेंबर|\d{4})",
    },
    "gu": {
        "PERSON":   r"[A-Z][a-z]+ [A-Z][a-z]+",
        "ORG":      r"(?:સરકાર|મંત્રાલય|સંસ્થા|પક્ષ|કંપની|સંગઠન|બોર્ડ|આયોગ)",
        "LOCATION": r"(?:ભારત|ગુજરાત|અમદાવાદ|સુરત|વડોદરા|રાજકોટ|ભાવનગર|જામનગર|દિલ્હી|મુંબઈ)",
        "DATE":     r"(?:જાન્યુઆરી|ફેબ્રુઆરી|માર્ચ|એપ્રિલ|મે|જૂન|જુલાઈ|ઑગસ્ટ|સપ્ટેમ્બર|ઑક્ટોબર|નવેમ્બર|ડિસેમ્બર|\d{4})",
    },
    "te": {
        "PERSON":   r"[A-Z][a-z]+ [A-Z][a-z]+",
        "ORG":      r"(?:ప్రభుత్వం|మంత్రిత్వ శాఖ|సంస్థ|పార్టీ|కంపెనీ|సంఘం|బోర్డు|కమిషన్)",
        "LOCATION": r"(?:భారత్|తెలంగాణ|ఆంధ్రప్రదేశ్|హైదరాబాద్|విశాఖపట్నం|విజయవాడ|గుంటూరు|నెల్లూరు|దిల్లీ|ముంబై)",
        "DATE":     r"(?:జనవరి|ఫిబ్రవరి|మార్చి|ఏప్రిల్|మే|జూన్|జులై|ఆగస్టు|సెప్టెంబర్|అక్టోబర్|నవంబర్|డిసెంబర్|\d{4})",
    },
}

# ── Wikidata fact-checking ─────────────────────────────────────
WD_URL    = "https://www.wikidata.org/w/api.php"
WD_CACHE: Dict[str, dict] = {}


def clean_text(text: str) -> str:
    """Mirror of training notebook clean_text()."""
    text = str(text).strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_entities(text: str, lang: str) -> List[Dict]:
    """
    Regex-based NER matching training notebook's entity extraction.
    Returns list of {text, type} dicts.
    """
    patterns = NER_PATTERNS.get(lang, NER_PATTERNS["hi"])
    seen     = set()
    entities = []

    for etype, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            word = match.group().strip()
            if word and word not in seen and len(word) > 1:
                seen.add(word)
                entities.append({"text": word, "type": etype})

    return entities


def compute_credibility_signal(text: str) -> float:
    """
    Heuristic credibility score in [0, 1].
    Mirrors training notebook's credibility_signal column.
    """
    text_lower = text.lower()

    # Sensationalist words (any language script — rough heuristic)
    sensational_indicators = [
        "shocking", "breaking", "viral", "exposed", "leaked",
        "सनसनी", "खुलासा", "वायरल", "सच्चाई", "धमाकेदार",
        "सनसनाटी", "व्हायरल", "खुलासा",
        "ચોંકાવનારો", "વાઈરલ", "ભ્રષ્ટ",
        "షాకింగ్", "వైరల్", "బహిర్గతం",
        # Additional fake news signals
        "जानें सच", "असली सच", "मीडिया नहीं बताएगा", "सरकार छुपा रही",
        "तुरंत शेयर", "सावधान", "अलर्ट", "बड़ा खुलासा", "हड़कंप",
        "चौंकाने वाला", "होश उड़ा", "बड़ी साजिश", "षड्यंत्र",
        "लाख करोड़", "करोड़ों", "अरबों",          # exaggerated numbers
        "अगले महीने", "कल से", "आज रात",          # urgency signals
        "100%", "गारंटी", "पक्का",                # certainty claims
    ]

    # Credible markers
    credible_indicators = [
        "according to", "reported by", "confirmed", "official",
        "अनुसार", "अधिकारिक", "रिपोर्ट",
        "अधिकृत", "अहवाल",
        "સત્તાવાર", "અહેવાલ",
        "అధికారిక", "నివేదిక",
        # Additional credible signals
        "प्रेस विज्ञप्ति", "आधिकारिक बयान", "सरकारी आंकड़े",
        "शोध", "अध्ययन", "विशेषज्ञ", "वैज्ञानिक",
        "संसद", "न्यायालय", "सुप्रीम कोर्ट",
    ]

    score = 0.5
    for word in sensational_indicators:
        if word in text_lower:
            score -= 0.05
    for word in credible_indicators:
        if word in text_lower:
            score += 0.05

    # Penalize ALL CAPS
    if len(re.findall(r"[A-Z]{4,}", text)) > 3:
        score -= 0.1

    return max(0.0, min(1.0, score))


def query_wikidata(entity_text: str, lang_code: str) -> dict:
    """Query Wikidata for entity verification (with cache)."""
    key = f"{entity_text}_{lang_code}"
    if key in WD_CACHE:
        return WD_CACHE[key]

    try:
        resp = requests.get(
            WD_URL,
            params={
                "action": "wbsearchentities",
                "format": "json",
                "language": lang_code,
                "search": entity_text,
                "limit": 1,
                "type": "item",
            },
            headers={"User-Agent": "FakeNewsDetector/1.0 (REVA academic)"},
            timeout=4,
        )
        data = resp.json()
        if data.get("search"):
            hit  = data["search"][0]
            info = {
                "id":          hit.get("id", ""),
                "label":       hit.get("label", entity_text),
                "description": hit.get("description", ""),
                "verified":    True,
            }
        else:
            info = {
                "id": "", "label": entity_text,
                "description": "Not found in Wikidata",
                "verified": False,
            }
        time.sleep(0.05)   # be polite to Wikidata
    except Exception:
        info = {
            "id": "", "label": entity_text,
            "description": "Query failed",
            "verified": False,
        }

    WD_CACHE[key] = info
    return info


def detect_language(text: str) -> str:
    """
    Script-based language detection (fast, no external deps).
    Falls back to langdetect if installed.
    """
    # Count chars in each script range
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    gujarati   = len(re.findall(r"[\u0A80-\u0AFF]", text))
    telugu     = len(re.findall(r"[\u0C00-\u0C7F]", text))

    if gujarati > devanagari and gujarati > telugu:
        return "gu"
    if telugu > devanagari:
        return "te"
    if devanagari > 0:
        # Marathi vs Hindi: check common Marathi words
        marathi_markers = re.findall(
            r"(?:आहे|नाही|होते|झाले|केले|सांगितले|मुंबई|पुणे|नागपूर)", text
        )
        if marathi_markers:
            return "mr"
        return "hi"

    # Fallback: try langdetect
    try:
        from langdetect import detect
        lang = detect(text)
        if lang in ("hi", "mr", "gu", "te"):
            return lang
    except Exception:
        pass

    return "hi"  # default
