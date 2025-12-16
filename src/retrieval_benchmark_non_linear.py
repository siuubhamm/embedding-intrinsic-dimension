import os, re, time, argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt


def tokenize(text: str):
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def load_abstracts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def build_needed_vocab(abstracts, max_unique_tokens=200000):
    toks = []
    for a in abstracts:
        toks.extend(tokenize(a))
    freq = Counter(toks)
    return set([t for t, _ in freq.most_common(max_unique_tokens)])


def load_glove_subset(glove_path: str, dim: int, needed_vocab: set):
    emb = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != dim + 1:
                continue
            w = parts[0]
            if w in needed_vocab:
                emb[w] = np.asarray(parts[1:], dtype=np.float32)
    return emb


def build_abstract_embeddings(abstracts, emb, dim: int):
    mats = []
    for a in abstracts:
        toks = tokenize(a)
        vecs = [emb[t] for t in toks if t in emb]
        if vecs:
            mats.append(np.mean(vecs, axis=0))
    if not mats:
        raise RuntimeError("No valid abstracts after embedding lookup.")
    return np.stack(mats).astype(np.float32)


def l2_normalize(X: np.ndarray):
    X = X.copy()
    faiss.normalize_L2(X)
    return X


def build_flat_ip_index(X: np.ndarray):
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    return index


def ndcg_at_k(I_true_R: np.ndarray, D_true_R: np.ndarray, I_pred_k: np.ndarray, k: int) -> np.ndarray:
    B, R = I_true_R.shape
    rel = D_true_R.astype(np.float32)
    mn = rel.min(axis=1, keepdims=True)
    mx = rel.max(axis=1, keepdims=True)
    rel = (rel - mn) / (mx - mn + 1e-12)

    gains = np.zeros((B, k), dtype=np.float32)
    for j in range(k):
        ids = I_pred_k[:, j]
        matches = (I_true_R == ids[:, None])
        found = matches.any(axis=1)
        idx = np.argmax(matches, axis=1)
        if np.any(found):
            gains[found, j] = rel[found, idx[found]]

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (gains * discounts[None, :]).sum(axis=1)

    rel_sorted = np.sort(rel, axis=1)[:, ::-1]
    ideal = rel_sorted[:, :k]
    idcg = (ideal * discounts[None, :]).sum(axis=1) + 1e-12

    return dcg / idcg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=str, required=True)
    ap.add_argument("--glove300", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs")

    ap.add_argument("--umap_dim", type=int, default=15)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--R_true", type=int, default=100)

    ap.add_argument("--num_queries", type=int, default=1_000_000)
    ap.add_argument("--batch_size", type=int, default=50_000)
    ap.add_argument("--query_noise", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=2025)

    ap.add_argument("--n_neighbors", type=int, default=15)
    ap.add_argument("--min_dist", type=float, default=0.1)
    ap.add_argument("--umap_metric", type=str, default="cosine")

    ap.add_argument("--cache_prefix", type=str, default="glove300_sample50k")
    args = ap.parse_args()

    try:
        import umap
    except Exception as e:
        raise RuntimeError("umap-learn is required: pip install umap-learn") from e

    os.makedirs(args.out_dir, exist_ok=True)

    ED = 300
    rng = np.random.default_rng(args.seed)

    cache_X = os.path.join(args.out_dir, f"X_abs_{args.cache_prefix}.npy")
    if os.path.exists(cache_X):
        X300 = np.load(cache_X)
    else:
        abstracts = load_abstracts(args.sample)
        vocab = build_needed_vocab(abstracts, max_unique_tokens=200000)
        emb = load_glove_subset(args.glove300, ED, vocab)
        X300 = build_abstract_embeddings(abstracts, emb, ED)
        np.save(cache_X, X300)

    X300 = l2_normalize(X300)
    n_db = X300.shape[0]

    index300 = build_flat_ip_index(X300)

    t0 = time.perf_counter()
    reducer = umap.UMAP(
        n_components=args.umap_dim,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
        transform_seed=args.seed,
        verbose=True,
    )
    X15 = reducer.fit_transform(X300)
    X15 = np.asarray(X15, dtype=np.float32)
    X15 = l2_normalize(X15)
    t1 = time.perf_counter()
    umap_build_sec = t1 - t0

    index15 = build_flat_ip_index(X15)

    total_search_300 = 0.0
    total_umap_q = 0.0
    total_search_15 = 0.0
    ndcg_sum_15 = 0.0
    processed = 0

    while processed < args.num_queries:
        B = min(args.batch_size, args.num_queries - processed)
        base_idx = rng.integers(0, n_db, size=B)
        Q300 = X300[base_idx] + args.query_noise * rng.standard_normal((B, ED)).astype(np.float32)
        Q300 = l2_normalize(Q300)

        t0 = time.perf_counter()
        D_true, I_true = index300.search(Q300, args.R_true)
        t1 = time.perf_counter()
        total_search_300 += (t1 - t0)

        t0 = time.perf_counter()
        Q15 = reducer.transform(Q300)
        Q15 = np.asarray(Q15, dtype=np.float32)
        Q15 = l2_normalize(Q15)
        t1 = time.perf_counter()
        total_umap_q += (t1 - t0)

        t0 = time.perf_counter()
        _, I15 = index15.search(Q15, args.topk)
        t1 = time.perf_counter()
        total_search_15 += (t1 - t0)

        nd = ndcg_at_k(I_true, D_true, I15, args.topk)
        ndcg_sum_15 += float(nd.sum())

        processed += B
        print(f"processed {processed}/{args.num_queries}")

    q = args.num_queries
    ndcg15 = ndcg_sum_15 / q

    us300 = 1e6 * (total_search_300 / q)
    us15_search = 1e6 * (total_search_15 / q)
    us15_total = 1e6 * ((total_umap_q + total_search_15) / q)

    speedup_search_only = (total_search_300 / total_search_15) if total_search_15 > 0 else np.nan
    speedup_end_to_end = (total_search_300 / (total_umap_q + total_search_15)) if (total_umap_q + total_search_15) > 0 else np.nan

    row = {
        "sample_file": Path(args.sample).name,
        "n_db": int(n_db),
        "ED": ED,
        "umap_dim": int(args.umap_dim),
        "topk": int(args.topk),
        "R_true": int(args.R_true),
        "num_queries": int(args.num_queries),
        "batch_size": int(args.batch_size),
        "query_noise": float(args.query_noise),

        "n_neighbors": int(args.n_neighbors),
        "min_dist": float(args.min_dist),
        "umap_metric": args.umap_metric,

        "umap_build_sec": float(umap_build_sec),
        "search300_us_per_query": float(us300),
        "umap_us_per_query": float(1e6 * (total_umap_q / q)),
        "search15_us_per_query": float(us15_search),
        "total15_us_per_query": float(us15_total),
        "speedup_search_only": float(speedup_search_only),
        "speedup_end_to_end": float(speedup_end_to_end),
        "ndcg_at_10_15d": float(ndcg15),
        "loss_15d": float(1.0 - ndcg15),
    }

    out_csv = os.path.join(args.out_dir, f"umap_ndcg_ed300_u{args.umap_dim}_k{args.topk}.csv")
    pd.DataFrame([row]).to_csv(out_csv, index=False)

    fig1 = os.path.join(args.out_dir, f"time_bars_umap_ed300_u{args.umap_dim}_k{args.topk}.png")
    plt.figure(figsize=(8, 6))
    labels = ["300D search", "15D search", "15D total (UMAP+search)"]
    vals = [us300, us15_search, us15_total]
    plt.bar(labels, vals)
    plt.ylabel("µs per query")
    plt.title("Per-query latency comparison (UMAP)")
    plt.tight_layout()
    plt.savefig(fig1, dpi=200)

    fig2 = os.path.join(args.out_dir, f"speedup_vs_loss_umap_ed300_u{args.umap_dim}_k{args.topk}.png")
    plt.figure(figsize=(8, 6))
    plt.scatter([speedup_end_to_end], [1.0 - ndcg15])
    plt.xlabel("Speedup (300D search / (UMAP+15D search))")
    plt.ylabel(f"Loss = 1 - nDCG@{args.topk}")
    plt.title("Speedup vs Accuracy Loss (UMAP)")
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)

    fig3 = os.path.join(args.out_dir, f"ndcg_vs_total_time_umap_ed300_u{args.umap_dim}_k{args.topk}.png")
    plt.figure(figsize=(8, 6))
    plt.scatter([us15_total], [ndcg15])
    plt.xlabel("15D total latency (µs/query) = UMAP + search")
    plt.ylabel(f"nDCG@{args.topk}")
    plt.title("Accuracy vs End-to-end Latency (UMAP)")
    plt.tight_layout()
    plt.savefig(fig3, dpi=200)

    print("Saved:", out_csv)
    print("Saved:", fig1)
    print("Saved:", fig2)
    print("Saved:", fig3)
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
