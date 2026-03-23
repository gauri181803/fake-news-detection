import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
from collections import OrderedDict

FEAT_DIM   = 769
HIDDEN_DIM = 128
OUT_DIM    = 2
DROPOUT    = 0.3

EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("article", "mentions",      "entity"),
    ("article", "supported_by",  "fact"),
    ("entity",  "mentioned_in",  "article"),
    ("entity",  "linked_to",     "fact"),
    ("fact",    "supports",      "entity"),
]
N_EDGES = len(EDGE_TYPES)

def _pyg_key(src, rel, dst):
    return f"<{src}___{rel}___{dst}>"


class HeteroConvLayer(nn.Module):
    PARAMS_PER_EDGE = 3

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        params = []
        for _ in EDGE_TYPES:
            params.append(nn.Parameter(torch.empty(out_dim, in_dim)))
            params.append(nn.Parameter(torch.zeros(out_dim)))
            params.append(nn.Parameter(torch.empty(out_dim, in_dim)))
        self.params = nn.ParameterList(params)
        for i in range(0, len(self.params), self.PARAMS_PER_EDGE):
            nn.init.kaiming_uniform_(self.params[i])
            nn.init.kaiming_uniform_(self.params[i + 2])

    def _sage(self, x_src, x_dst, ei, ll_w, ll_b, lr_w):
        out   = F.linear(x_dst, ll_w, ll_b)
        N_dst = x_dst.size(0)
        if ei.size(1) > 0:
            si, di = ei[0], ei[1]
            agg = torch.zeros(N_dst, x_src.size(1), device=x_src.device, dtype=x_src.dtype)
            cnt = torch.zeros(N_dst, 1,              device=x_src.device, dtype=x_src.dtype)
            agg.scatter_add_(0, di.unsqueeze(1).expand(-1, x_src.size(1)), x_src[si])
            cnt.scatter_add_(0, di.unsqueeze(1),
                             torch.ones(ei.size(1), 1, device=x_src.device, dtype=x_src.dtype))
            agg = agg / cnt.clamp(min=1)
        else:
            agg = torch.zeros(N_dst, x_src.size(1), device=x_src.device, dtype=x_src.dtype)
        out = out + F.linear(agg, lr_w)
        return F.normalize(out, p=2, dim=-1)

    def forward(self, x_dict, edge_index_dict):
        out = {}
        for i, (src_t, rel, dst_t) in enumerate(EDGE_TYPES):
            if src_t not in x_dict or dst_t not in x_dict:
                continue
            ek = (src_t, rel, dst_t)
            if ek not in edge_index_dict:
                continue
            base = i * self.PARAMS_PER_EDGE
            h = self._sage(x_dict[src_t], x_dict[dst_t], edge_index_dict[ek],
                           self.params[base], self.params[base+1], self.params[base+2])
            out[dst_t] = out[dst_t] + h if dst_t in out else h
        for ntype, feat in x_dict.items():
            if ntype not in out:
                out[ntype] = torch.zeros(feat.size(0), self.out_dim,
                                         device=feat.device, dtype=feat.dtype)
        return out


def _remap_key(pyg_key: str):
    if pyg_key.startswith("head."):
        return pyg_key
    if "aggr_module" in pyg_key:
        return None
    for layer in ("conv1", "conv2"):
        if not pyg_key.startswith(f"{layer}.convs."):
            continue
        for i, (src, rel, dst) in enumerate(EDGE_TYPES):
            if _pyg_key(src, rel, dst) not in pyg_key:
                continue
            base = i * HeteroConvLayer.PARAMS_PER_EDGE
            if   "lin_l.weight" in pyg_key: idx = base
            elif "lin_l.bias"   in pyg_key: idx = base + 1
            elif "lin_r.weight" in pyg_key: idx = base + 2
            else: return None
            return f"{layer}.params.{idx}"
    return None


class HeteroRGCN(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, hidden=HIDDEN_DIM, out_dim=OUT_DIM, dropout=DROPOUT):
        super().__init__()
        self.conv1 = HeteroConvLayer(in_dim, hidden)
        self.conv2 = HeteroConvLayer(hidden, hidden)
        self.head  = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )
        self.drop = dropout

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = {k: F.dropout(F.relu(v), self.drop, self.training) for k, v in x.items()}
        x = self.conv2(x, edge_index_dict)
        x = {k: F.relu(v) for k, v in x.items()}
        logits = self.head(x["article"])
        return logits, x

    @classmethod
    def load(cls, path: str, device=torch.device("cpu")) -> "HeteroRGCN":
        pyg_sd = torch.load(path, map_location=device, weights_only=True)
        model  = cls().to(device)
        new_sd = OrderedDict()
        for k, v in pyg_sd.items():
            mk = _remap_key(k)
            if mk is not None:
                new_sd[mk] = v
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        print(f"   Loaded {len(new_sd)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"   Missing: {missing[:4]}")
        return model