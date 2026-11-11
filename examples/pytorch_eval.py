"""fastauc in a PyTorch evaluation loop.

    python examples/pytorch_eval.py
"""
import numpy as np
import torch

import fastauc


@torch.no_grad()
def evaluate(model, loader):
    """Collect predictions on CPU, then compute everything in one call."""
    probs, labels, queries = [], [], []
    for x, y, q in loader:
        probs.append(torch.sigmoid(model(x)).squeeze(-1).cpu())
        labels.append(y.cpu())
        queries.append(q.cpu())
    p = torch.cat(probs).numpy()
    y = torch.cat(labels).numpy()
    q = torch.cat(queries).numpy()

    out = fastauc.binary_metrics(y, p)                        # AUC, KS, log loss, ECE, ...
    out.update(fastauc.ranking_metrics(y, p, q, ks=(1, 10)))  # per-query NDCG, MRR, GAUC, ...
    boot = fastauc.bootstrap_roc_pr_auc(y, p, groups=q, n_bootstraps=500)
    out["roc_auc_ci95"] = boot.ci("roc_auc")                  # resamples whole queries
    return out


def main():
    # If you call fastauc from DataLoader workers or other parallel code:
    # fastauc.set_num_threads(1)
    torch.manual_seed(0)
    n, n_queries = 20_000, 1_000
    x = torch.randn(n, 8)
    y = (torch.rand(n) < torch.sigmoid(x[:, 0] * 1.5)).long()
    q = torch.randint(0, n_queries, (n,))
    model = torch.nn.Linear(8, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(100):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1), y.float()).backward()
        opt.step()

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y, q), batch_size=4096)
    metrics = evaluate(model, loader)
    for key in ("roc_auc", "roc_auc_ci95", "ks", "normalized_entropy", "ece", "ndcg@10", "gauc"):
        print(f"{key:>20}: {np.round(metrics[key], 4)}")
    assert metrics["roc_auc"] > 0.75 and metrics["normalized_entropy"] < 1.0


if __name__ == "__main__":
    main()
