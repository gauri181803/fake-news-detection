"""
FakeNewsPredictor
Loads trained HeteroRGCN checkpoints and runs inference on a single article
by constructing a minimal 3-node heterogeneous graph at inference time.
"""

import os
import re
import time
import asyncio
import numpy as np
import torch
from typing import Optional, List, Dict, Tuple
from app.classifier import FakeNewsClassifier, load_classifier, CLASSIFIER_PATH
from transformers import AutoTokenizer, AutoModel

from app.model import HeteroRGCN, FEAT_DIM, EDGE_TYPES
from app.preprocessing import (
    clean_text,
    detect_language,
    extract_entities,
    compute_credibility_signal,
    query_wikidata,
    LANG_META,
)
from app.schemas import PredictResponse, EntityEvidence

# ── Paths ──────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

LANG_MODEL_FILES: Dict[str, str] = {
    "hi": "hetero_rgcn_model_G.pth",       # Hindi checkpoint
    "mr": "hetero_rgcn_model_marathi.pth",
    "gu": "hetero_rgcn_model_G.pth",       # Gujarati (same file name from notebook)
    "te": "hetero_rgcn_model_telugu.pth",
}

XLM_R_MODEL = "xlm-roberta-base"
MAX_LEN      = 128
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENTITY_TYPE_MAP = {"PERSON": 0, "ORG": 1, "LOCATION": 2, "DATE": 3}


class FakeNewsPredictor:
    def __init__(self):
        self.tokenizer     = None
        self.xlm_model     = None
        self.loaded_models: Dict[str, HeteroRGCN] = {}
        self.classifier = None

    # ── Load XLM-R embedder ────────────────────────────────────
    def _load_embedder(self):
        if self.tokenizer is not None:
            return
        print(f"🤖  Loading {XLM_R_MODEL}...")
        self.tokenizer = AutoTokenizer.from_pretrained(XLM_R_MODEL)
        self.xlm_model = AutoModel.from_pretrained(XLM_R_MODEL).to(DEVICE)
        self.xlm_model.eval()
        print("✅  XLM-R loaded.")

    # ── Load one GNN checkpoint ────────────────────────────────
    def _load_gnn(self, lang: str):
        if lang in self.loaded_models:
            return
        fname = LANG_MODEL_FILES.get(lang)
        if not fname:
            raise ValueError(f"No model file configured for language: {lang}")
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"⚠️   Model file not found: {path}. Running in XLM-R-only mode for {lang}.")
            self.loaded_models[lang] = None
            return
        model = HeteroRGCN.load(path, device=DEVICE)
        model.eval()
        self.loaded_models[lang] = model
        print(f"✅  Loaded GNN for [{lang}] from {fname}")

    def load_all_models(self):
        """Pre-load XLM-R and all language GNNs at startup."""
        self._load_embedder()
        self.classifier = load_classifier()
        for lang in LANG_MODEL_FILES:
            self._load_gnn(lang)

    # ── Embed a single text ────────────────────────────────────
    @torch.no_grad()
    def _embed(self, text: str) -> np.ndarray:
        enc  = self.tokenizer(
            text, padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt"
        )
        ids  = enc["input_ids"].to(DEVICE)
        mask = enc["attention_mask"].to(DEVICE)
        out  = self.xlm_model(input_ids=ids, attention_mask=mask)
        m    = mask.unsqueeze(-1).float()
        emb  = ((out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9))
        return emb.squeeze(0).cpu().numpy()

    # ── Build a minimal inference graph ───────────────────────
    def _build_inference_graph(
        self,
        embedding: np.ndarray,
        credibility: float,
        entities: List[Dict],
        verified_facts: List[Dict],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple, torch.Tensor]]:

        # ── Article node (always index 0) ──
        art_feat = torch.tensor(
            np.concatenate([embedding, [credibility]]),
            dtype=torch.float,
        ).unsqueeze(0)  # (1, 769)

        # ── Entity nodes ──
        N_ENT = max(len(entities), 1)
        ent_oh = np.zeros((N_ENT, 4), dtype=np.float32)
        for i, ent in enumerate(entities):
            ent_oh[i, ENTITY_TYPE_MAP.get(ent["type"], 0)] = 1.0
        ent_rand  = np.random.randn(N_ENT, FEAT_DIM - 4).astype(np.float32) * 0.01
        ent_feats = torch.tensor(np.concatenate([ent_oh, ent_rand], axis=1), dtype=torch.float)

        # ── Fact nodes ──
        N_FACT = max(len(verified_facts), 1)
        fact_ver = np.array(
            [[float(f["verified"])] for f in verified_facts], dtype=np.float32
        ) if verified_facts else np.zeros((1, 1), dtype=np.float32)
        fact_rand  = np.random.randn(N_FACT, FEAT_DIM - 1).astype(np.float32) * 0.01
        fact_feats = torch.tensor(np.concatenate([fact_ver, fact_rand], axis=1), dtype=torch.float)

        # ── Edges — all indices must be within bounds ──
        # article(0) → each entity(i)
        ae_src = [0] * N_ENT          # article index is always 0
        ae_dst = list(range(N_ENT))   # entity indices 0..N_ENT-1

        # article(0) → each fact(i)
        af_src = [0] * N_FACT
        af_dst = list(range(N_FACT))

        # entity(i) → article(0)
        ea_src = list(range(N_ENT))
        ea_dst = [0] * N_ENT

        # entity(i) → fact(i)  only for min(N_ENT, N_FACT) pairs
        N_LINK = min(N_ENT, N_FACT)
        ef_src = list(range(N_LINK))
        ef_dst = list(range(N_LINK))

        # fact(i) → entity(i)
        fe_src = list(range(N_LINK))
        fe_dst = list(range(N_LINK))

        def mk_ei(src, dst):
            if not src:
                return torch.zeros((2, 0), dtype=torch.long)
            return torch.tensor([src, dst], dtype=torch.long)

        x_dict = {
            "article": art_feat,
            "entity":  ent_feats,
            "fact":    fact_feats,
        }
        edge_index_dict = {
            ("article", "mentions",      "entity"):  mk_ei(ae_src, ae_dst),
            ("article", "supported_by",  "fact"):    mk_ei(af_src, af_dst),
            ("entity",  "mentioned_in",  "article"): mk_ei(ea_src, ea_dst),
            ("entity",  "linked_to",     "fact"):    mk_ei(ef_src, ef_dst),
            ("fact",    "supports",      "entity"):  mk_ei(fe_src, fe_dst),
        }
        return x_dict, edge_index_dict
    # ── Main prediction entry point ────────────────────────────
    async def predict(
        self,
        text: str,
        language: Optional[str] = None,
    ) -> PredictResponse:

        # 1. Language detection
        lang = language if language and language in LANG_MODEL_FILES else detect_language(text)
        meta = LANG_META[lang]

        # 2. Clean text
        clean = clean_text(text)

        # 3. Embed
        loop      = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._embed, clean)

        # 4. Credibility heuristic
        credibility = compute_credibility_signal(clean)

        # 5. NER
        entities = extract_entities(clean, lang)

        # 6. Wikidata verification
        verified_facts: List[Dict] = []
        entity_evidences: List[EntityEvidence] = []

        wd_lang = meta["wikidata_code"]
        for ent in entities[:3]:
            wk = await loop.run_in_executor(
                None, query_wikidata, ent["text"], wd_lang
            )
            verified_facts.append({"entity": ent["text"], "verified": wk["verified"]})
            entity_evidences.append(
                EntityEvidence(
                    entity=ent["text"],
                    entity_type=ent["type"],
                    wikidata_verified=wk["verified"],
                    wikidata_description=wk.get("description"),
                )
            )

        verified_count   = sum(1 for e in entity_evidences if e.wikidata_verified)
        unverified_count = len(entity_evidences) - verified_count

        # 7. Build graph
        x_dict, edge_index_dict = self._build_inference_graph(
            embedding, credibility, entities, verified_facts
        )
        x_dict          = {k: v.to(DEVICE) for k, v in x_dict.items()}
        edge_index_dict = {k: v.to(DEVICE) for k, v in edge_index_dict.items()}

        # 8. Inference
        gnn = self.loaded_models.get(lang)

        if self.classifier is not None:
            emb_tensor = torch.tensor(embedding, dtype=torch.float).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = self.classifier(emb_tensor)
                probs  = torch.softmax(logits[0], dim=0).cpu().numpy()

        elif gnn is not None:
            with torch.no_grad():
                logits, _ = gnn(x_dict, edge_index_dict)
                probs = torch.softmax(logits[0], dim=0).cpu().numpy()

        else:
            fs    = max(0.0, min(1.0, 1.0 - credibility))
            probs = np.array([1.0 - fs, fs])

        # 9. Build response
        real_score = float(probs[0])
        fake_score = float(probs[1])
        label      = "FAKE" if fake_score > real_score else "REAL"
        confidence = max(real_score, fake_score)

        if label == "FAKE":
            explanation = (
                f"The article was classified as FAKE with {confidence:.1%} confidence. "
                f"{verified_count} of {len(entity_evidences)} entities were verified in Wikidata. "
                f"Credibility signal score: {credibility:.2f}."
            )
        else:
            explanation = (
                f"The article was classified as REAL with {confidence:.1%} confidence. "
                f"{verified_count} of {len(entity_evidences)} entities were verified in Wikidata. "
                f"Credibility signal score: {credibility:.2f}."
            )

        return PredictResponse(
            prediction=label,
            confidence=round(confidence, 4),
            fake_score=round(fake_score, 4),
            real_score=round(real_score, 4),
            language_detected=lang,
            language_name=meta["name"],
            entities_found=entity_evidences,
            verified_count=verified_count,
            unverified_count=unverified_count,
            explanation=explanation,
        )