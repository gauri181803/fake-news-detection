# Multilingual Fake News Detection — Backend API

**Tech Stack:** FastAPI · XLM-RoBERTa · Heterogeneous R-GCN · Wikidata · PyTorch Geometric

Supports **Hindi, Marathi, Gujarati, Telugu**.

---

## Project Structure

```
fakenews_backend/
├── app/
│   ├── main.py           # FastAPI app, routes
│   ├── predictor.py      # Model loading + inference logic
│   ├── model.py          # HeteroRGCN architecture (must match training)
│   ├── preprocessing.py  # NER, language detection, Wikidata queries
│   └── schemas.py        # Pydantic request/response models
├── models/               # ← Place your .pth files here
│   ├── hetero_rgcn_model_G.pth        ( Gujarati)
│   ├── hetero_rgcn_model_marathi.pth
│   └── hetero_rgcn_model_telugu.pth
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Setup

### 1. Place Model Files
Copy your trained `.pth` files into the `models/` directory:
```bash
cp hetero_rgcn_model_G.pth        models/
cp hetero_rgcn_model_marathi.pth  models/
cp hetero_rgcn_model_telugu.pth   models/
```

### 2. Install Dependencies
```bash
pip install torch==2.3.0
pip install torch-geometric==2.5.3
pip install torch-scatter torch-sparse \
  --find-links https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install -r requirements.txt
```

### 3. Run
```bash
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the Swagger UI.

---

## Docker

```bash
# Build
docker build -t fakenews-api .

# Run
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  fakenews-api
```

---

## API Reference

### `POST /predict`

```json
Request:
{
  "text": "नरेंद्र मोदी ने आज नई दिल्ली में बैठक की।",
  "language": "hi"        // optional — auto-detected if omitted
}

Response:
{
  "prediction":        "REAL",
  "confidence":        0.9743,
  "fake_score":        0.0257,
  "real_score":        0.9743,
  "language_detected": "hi",
  "language_name":     "Hindi",
  "entities_found": [
    {
      "entity":               "नरेंद्र मोदी",
      "entity_type":          "PERSON",
      "wikidata_verified":    true,
      "wikidata_description": "Prime Minister of India"
    }
  ],
  "verified_count":   1,
  "unverified_count": 0,
  "explanation": "The article was classified as REAL with 97.4% confidence..."
}
```

### `GET /health`
Returns loaded model names and supported language codes.

---

## Inference Pipeline

```
Input Text
    │
    ▼
Language Detection (script-based + langdetect fallback)
    │
    ▼
Text Cleaning  (remove URLs, mentions, whitespace)
    │
    ▼
XLM-R Embedding  (768-dim mean pooled)
    │
    ▼
Credibility Signal  (heuristic score → dim 769)
    │
    ▼
NER  (regex-based, per-language patterns)
    │
    ▼
Wikidata Entity Verification  (parallel async queries)
    │
    ▼
Heterogeneous Graph Construction
  [article] ─mentions─▶ [entity] ─linked_to─▶ [fact]
    │
    ▼
HeteroRGCN Inference  (2-layer SAGEConv + MLP head)
    │
    ▼
Prediction: FAKE / REAL + Confidence + Evidence
```

---

## Model Performance (from training notebooks)

| Language  | Test Accuracy | ROC-AUC | F1-Macro |
|-----------|:-------------:|:-------:|:--------:|
| Hindi     | 98.28%        | 0.9976  | 0.9827   |


---

## Environment Variables

| Variable    | Default   | Description                        |
|-------------|-----------|------------------------------------|
| `MODEL_DIR` | `models/` | Path to folder containing .pth files |

---

