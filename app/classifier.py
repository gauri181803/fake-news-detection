"""
Lightweight XLM-R based binary classifier for single-article inference.
Fine-tuned on top of the same XLM-R embeddings used in GNN training.
No hardcoded keywords — pure contextual understanding.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel

CLASSIFIER_PATH = os.environ.get("MODEL_DIR", "models") + "/xlmr_classifier.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FakeNewsClassifier(nn.Module):
    """
    Simple MLP head on top of frozen XLM-R embeddings.
    Input: 768-dim XLM-R embedding
    Output: 2-class logits (Real, Fake)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


def get_embedding(text: str, tokenizer, xlm_model) -> np.ndarray:
    enc  = tokenizer(text, padding=True, truncation=True,
                     max_length=128, return_tensors="pt")
    ids  = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        out = xlm_model(input_ids=ids, attention_mask=mask)
    m   = mask.unsqueeze(-1).float()
    emb = ((out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9))
    return emb.squeeze(0).cpu().numpy()


def train_classifier(texts, labels, tokenizer, xlm_model, epochs=10):
    """
    Train the classifier on a list of texts and binary labels (0=Real, 1=Fake).
    """
    print(f"🔄 Generating embeddings for {len(texts)} samples...")
    embeddings = np.array([get_embedding(t, tokenizer, xlm_model) for t in texts])

    X = torch.tensor(embeddings, dtype=torch.float).to(DEVICE)
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)

    model = FakeNewsClassifier().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()

    print(f"🚀 Training classifier for {epochs} epochs...")
    model.train()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        out  = model(X)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        acc = (out.argmax(1) == y).float().mean().item()
        if ep % 2 == 0:
            print(f"   Ep {ep:>3} | Loss: {loss.item():.4f} | Acc: {acc:.1%}")

    torch.save(model.state_dict(), CLASSIFIER_PATH)
    print(f"✅ Classifier saved to {CLASSIFIER_PATH}")
    return model


def load_classifier():
    """Load trained classifier if it exists."""
    if not os.path.exists(CLASSIFIER_PATH):
        return None
    model = FakeNewsClassifier().to(DEVICE)
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    model.eval()
    print("✅ XLM-R classifier loaded.")
    return model