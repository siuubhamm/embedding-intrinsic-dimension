from __future__ import annotations

import re
from collections import Counter

import numpy as np
import faiss


def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def load_glove_txt(path, dim):
    vecs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != dim + 1:
                continue
            word = parts[0]
            v = np.asarray(parts[1:], dtype=np.float32)
            vecs[word] = v
    return vecs


def build_unique_token_matrix(abstracts, emb, max_unique = 50_000):
    tokens = []
    for a in abstracts:
        tokens.extend(tokenize(a))
    freq = Counter(tokens)
    unique_tokens = [t for t, _ in freq.most_common(max_unique) if t in emb]

    mat = [emb[t] for t in unique_tokens]
    if len(mat) == 0:
        return None, unique_tokens
    return np.stack(mat).astype(np.float32), unique_tokens


def build_abstract_matrix(abstracts, emb):
    mats = []
    for a in abstracts:
        toks = tokenize(a)
        vecs = [emb[t] for t in toks if t in emb]
        if vecs:
            mats.append(np.mean(vecs, axis=0))
    if len(mats) == 0:
        return None
    return np.stack(mats).astype(np.float32)


def faiss_knn_radii(X, k):
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    D, _ = index.search(X, k + 1)
    D = D[:, 1:]
    R = np.sqrt(np.maximum(D, 1e-12))
    return R


def lid_mle_safe(R, eps = 1e-8):
    mask = R[:, -1] > eps
    if not np.any(mask):
        return np.array([], dtype=np.float32)

    R2 = R[mask]
    rk = R2[:, -1:]
    logs = np.log(rk / np.maximum(R2[:, :-1], eps))
    lids = (np.mean(logs, axis=1)) ** (-1)
    return lids.astype(np.float32)


def global_id(lids):
    return float(1.0 / np.mean(1.0 / lids))
