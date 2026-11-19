"""
diagnose_pvcs_samples.py
=======================
Verify what "edge / middle / representative" actually are, on the exact logit-gradient embedding that PVCS uses.

Idea: on the warm-up model, reproduce PVCS_pairwise's coreset-selection step exactly, label each selected point as edge / middle / random, then compute for each group:
  - ‖g‖  (logit-gradient norm  = gradient magnitude ~ difficulty)
  - loss (cross-entropy)       = -log(p_y)
  - acc  (classification accuracy within the group)
compared against the whole set (ALL) as a reference.

KEY POINT: the embedding is g = p - e_y (logit-gradient). Hence no forward pass is needed:
  p_y   = 1 + g[y]        (the true-class component of g is p_y - 1)
  loss  = -log(p_y)
  pred  = argmax(g + e_y)
The script checks that p = g + e_y is a valid distribution (sum ~1, within [0,1]) to make
sure the embedding has the form p - e_y; otherwise it prints a warning.

Default run over all 4 datasets (checkpoints follow results_<ds>/checkpoints/warmup_seed<seed>.pt):
  python diagnose_pvcs_points.py --seed 42 --coreset_fraction 0.1 --pvcs_edge_ratio 0.5
  # or run a subset:  --datasets cifar10 svhn
"""
import os
import argparse
import numpy as np
from numpy.linalg import norm
import torch

from utils import (load_cifar10_all, load_cifar100_all, load_svhn_all,
                   load_fashion_mnist_all, get_indices_by_class,
                   compute_gradient_representations)
from model_resnet import resnet20
from pvcs_coreset import BVS, _build_nearest_class_pairs


# --------------------------------------------------------------------------------------
# Configuration for the 4 datasets (same as pvcs_prop_verify.py)
# --------------------------------------------------------------------------------------
DATASETS = ["cifar10", "cifar100", "svhn", "fashionmnist"]
LOADERS = {
    "cifar10": load_cifar10_all,
    "cifar100": load_cifar100_all,
    "svhn": load_svhn_all,
    "fashionmnist": load_fashion_mnist_all,
}
NCLASS_BY_DS = {"cifar10": 10, "cifar100": 100, "svhn": 10, "fashionmnist": 10}
M_BY_DS = {"cifar10": 2, "svhn": 2, "fashionmnist": 2, "cifar100": 4}
CKPT_DIR = {"cifar10": "results_cifar10", "cifar100": "results_cifar100",
            "svhn": "results_svhn", "fashionmnist": "results_mnist"}
PRETTY = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
          "svhn": "SVHN", "fashionmnist": "FashionMNIST"}


def build_model_for(ds, nclass, device):
    """Build the right architecture per dataset: ResNet-20 for color-image datasets, LogisticRegression for FashionMNIST."""
    if ds == "fashionmnist":
        from model_logistic_regression import LogisticRegression
        return LogisticRegression(784, nclass).to(device)
    return resnet20(num_classes=nclass).to(device)


# --------------------------------------------------------------------------------------
# 1) Compute X_all (logit-gradient) exactly as select_pvcs_coreset does
# --------------------------------------------------------------------------------------
def compute_Xall(model, dataset, indices_by_class, device, gradient_type, bs):
    classes = [c for c in indices_by_class if len(indices_by_class[c]) > 0]
    X_blocks, lbl_blocks, global_ref = [], [], []
    for c in classes:
        ci = indices_by_class[c]
        gr = compute_gradient_representations(
            model=model, full_dataset=dataset, indices=ci,
            device=device, gradient_type=gradient_type, batch_size=bs)
        Xc = gr.detach().cpu().numpy().astype(np.float64)
        X_blocks.append(Xc)
        lbl_blocks.append(np.full(len(ci), c))
        global_ref.extend(list(ci))
    X_all = np.vstack(X_blocks)
    labels_all = np.concatenate(lbl_blocks)
    return X_all, labels_all, np.array(global_ref)


# --------------------------------------------------------------------------------------
# 2) Reproduce PVCS_pairwise BUT return the edge / middle / random index sets (in X_all indexing).
#    Logic identical to PVCS_pairwise in pvcs_coreset.py (fill_mode="random").
# --------------------------------------------------------------------------------------
def pvcs_label_sets(X, labels, target_ratio, edge_ratio, M, random_state):
    n = X.shape[0]
    classes = np.unique(labels)
    idx_by_class = {c: np.where(labels == c)[0] for c in classes}

    total_target = max(1, int(np.ceil(target_ratio * n)))
    sizes = {c: len(idx_by_class[c]) for c in classes}
    tot = sum(sizes.values())
    allocated = {c: int(total_target * sizes[c] / tot) for c in classes}
    rem = total_target - sum(allocated.values())
    for c in sorted(allocated, key=lambda x: -((total_target * sizes[x] / tot) - allocated[x])):
        if rem <= 0:
            break
        allocated[c] += 1
        rem -= 1

    centroids = {c: X[idx_by_class[c]].mean(axis=0) for c in classes}
    mn = min(M, len(classes) - 1)
    pairs = _build_nearest_class_pairs(centroids, mn)
    ppc = {c: 0 for c in classes}
    for (i, j) in pairs:
        ppc[i] += 1
        ppc[j] += 1

    selected = {c: set() for c in classes}
    edge = {c: set() for c in classes}
    middle = {c: set() for c in classes}
    for (i, j) in pairs:
        pidx = np.concatenate([idx_by_class[i], idx_by_class[j]])
        Xp = X[pidx]
        lp = labels[pidx]
        pn = len(pidx)
        si = allocated[i] / ppc[i] if ppc[i] > 0 else 0
        sj = allocated[j] / ppc[j] if ppc[j] > 0 else 0
        ptc = max(1, int(np.ceil(si + sj)))
        pr = min(1.0, ptc / pn)
        sel_local, e_local, m_local = BVS(Xp, lp, target_ratio=pr, edge_ratio=edge_ratio, return_stats=True)
        for gi in pidx[sel_local]:
            selected[labels[gi]].add(int(gi))
        for li in e_local:
            gi = int(pidx[li]); c = labels[gi]
            if gi not in middle[c]:
                edge[c].add(gi)
        for li in m_local:
            gi = int(pidx[li]); c = labels[gi]
            if gi not in edge[c]:
                middle[c].add(gi)

    rng = np.random.default_rng(random_state)
    random_set = {c: set() for c in classes}
    for c in classes:
        b = allocated[c]
        if b <= 0:
            continue
        sel = list(selected[c])
        if len(sel) < b:
            ss = set(sel)
            cand = [i for i in idx_by_class[c] if i not in ss]
            rng.shuffle(cand)
            chosen = cand[:b - len(sel)]
            random_set[c] = set(int(x) for x in chosen)
        if len(sel) > b:
            trim = set(sel[b:])
            edge[c] -= trim
            middle[c] -= trim

    def flat(d):
        out = set()
        for s in d.values():
            out |= s
        return out
    return flat(edge), flat(middle), flat(random_set)


# --------------------------------------------------------------------------------------
# 3) Compute ‖g‖ / loss / acc for an index set (DIRECTLY from the embedding g = p - e_y)
# --------------------------------------------------------------------------------------
def group_stats(S, X, labels):
    S = np.array(sorted(S), dtype=int)
    if len(S) == 0:
        return dict(n=0)
    g = X[S]
    y = labels[S].astype(int)
    gnorm = np.linalg.norm(g, axis=1)
    # p = g + e_y  -> p_y = 1 + g[y]
    P = g.copy()
    P[np.arange(len(S)), y] += 1.0
    py = np.clip(P[np.arange(len(S)), y], 1e-12, 1.0)
    loss = -np.log(py)
    acc = 100.0 * (P.argmax(axis=1) == y).mean()
    return dict(n=len(S),
                gnorm_mean=gnorm.mean(), gnorm_std=gnorm.std(),
                loss_mean=loss.mean(), loss_std=loss.std(), acc=acc,
                P=P)


def validate_embedding(X, labels):
    """Check that p = g + e_y is a valid distribution (sum ~1, within [0,1])."""
    y = labels.astype(int)
    P = X.copy()
    P[np.arange(len(X)), y] += 1.0
    row_sum = P.sum(axis=1)
    ok_sum = np.allclose(row_sum, 1.0, atol=1e-2)
    ok_range = (P.min() > -1e-2) and (P.max() < 1.0 + 1e-2)
    return ok_sum, ok_range, row_sum.mean(), P.min(), P.max()


# --------------------------------------------------------------------------------------
# 4) Run the diagnostic for ONE dataset (idea & printed table unchanged, only a dataset header added)
# --------------------------------------------------------------------------------------
def run_one_dataset(ds, args, device):
    nclass = NCLASS_BY_DS[ds]
    M = args.pvcs_max_neighbor_classes if args.pvcs_max_neighbor_classes is not None else M_BY_DS[ds]
    ckpt_path = os.path.join(CKPT_DIR[ds], "checkpoints", f"warmup_seed{args.seed}.pt")

    print("\n" + "=" * 92)
    print(f">>> {PRETTY[ds]}  (classes={nclass}, M={M}, frac={args.coreset_fraction}, r={args.pvcs_edge_ratio})")
    print("=" * 92)

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    trainset_full, _ = LOADERS[ds]()
    indices_by_class = get_indices_by_class(trainset_full, num_classes=nclass)

    model = build_model_for(ds, nclass, device)
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)  # our own checkpoint -> trusted
        state = ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck
        model.load_state_dict(state)
        print(f">>> Loaded model from: {ckpt_path}")
    else:
        print(f"!! Checkpoint {ckpt_path} not found -> using a RANDOMLY INITIALIZED model "
              f"(smoke test only; the numbers are meaningless). Please provide a real checkpoint.")
    model.eval()

    print(">>> Computing logit-gradient embedding (X_all)...")
    X_all, labels_all, _ = compute_Xall(model, trainset_full, indices_by_class,
                                        device, args.gradient_type, args.batch_size)
    print(f">>> X_all shape = {X_all.shape}")

    ok_sum, ok_range, msum, pmin, pmax = validate_embedding(X_all, labels_all)
    print(f">>> Embedding check g = p - e_y: sum p≈1? {ok_sum} (mean sum={msum:.4f}); "
          f"p in [0,1]? {ok_range} (min={pmin:.4f}, max={pmax:.4f})")
    if not (ok_sum and ok_range):
        print("   WARNING: the embedding does NOT look like p - e_y (scaled?). "
              "The derived loss/acc may be wrong — report back to switch to a direct forward pass.")

    E, Mid, R = pvcs_label_sets(X_all, labels_all, args.coreset_fraction,
                                args.pvcs_edge_ratio, M, args.seed)
    ALL = set(range(len(X_all)))
    SEL = E | Mid | R

    print(f"\n=== DIFFICULTY BY GROUP — {PRETTY[ds]} (logit-gradient embedding, first reselection round) ===")
    print(f"{'group':>10}{'n':>8}{'‖g‖ mean':>12}{'‖g‖ std':>10}{'loss mean':>12}{'loss std':>10}{'acc(%)':>9}")
    for name, S in [('EDGE', E), ('MIDDLE', Mid), ('RANDOM', R), ('SELECTED', SEL), ('ALL(full)', ALL)]:
        st = group_stats(S, X_all, labels_all)
        if st['n'] == 0:
            print(f"{name:>10}{0:>8}   (empty)")
            continue
        print(f"{name:>10}{st['n']:>8}{st['gnorm_mean']:>12.4f}{st['gnorm_std']:>10.4f}"
              f"{st['loss_mean']:>12.4f}{st['loss_std']:>10.4f}{st['acc']:>9.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--coreset_fraction', type=float, default=0.1)
    ap.add_argument('--pvcs_edge_ratio', type=float, default=0.5)
    ap.add_argument('--pvcs_max_neighbor_classes', type=int, default=None,
                    help='Override M for all datasets; None = use per-dataset M (M_BY_DS).')
    ap.add_argument('--gradient_type', type=str, default='logit')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--datasets', type=str, nargs='+', default=DATASETS,
                    help='List of datasets to run (default: all 4).')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}  seed={args.seed}  frac={args.coreset_fraction}  r={args.pvcs_edge_ratio}")

    for ds in args.datasets:
        if ds not in LOADERS:
            print(f"!! Skipping unsupported dataset: {ds}")
            continue
        run_one_dataset(ds, args, device)

    print("\nPrediction if 'edge' are truly HARD points (not easy representatives):")
    print("  EDGE has the HIGHEST ‖g‖ & loss and the LOWEST acc; MIDDLE (old definition) has low ‖g‖ (mixed with easy points);")
    print("  RANDOM approximates ALL (a representative, distributed sample).")


if __name__ == '__main__':
    main()