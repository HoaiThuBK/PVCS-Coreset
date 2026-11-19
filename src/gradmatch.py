"""
---file gradmatch.py---
Coreset selection with the GradMatch algorithm (Killamsetty et al., 2021).
"""
import time
import torch
from tqdm import tqdm
from utils import compute_gradient_representations


def _omp_reg(A, b, nnz, lam=1.0, positive=True, tol=1e-4, desc="GradMatch"):
    """
    Regularized Orthogonal Matching Pursuit, fitting b with a combination of the columns of A.
    
    :param A: Tensor (d, m) — each column is the gradient of one candidate.
    :param b: Tensor (d,) — the target vector to fit, usually the sum of the class's gradients.
    :param nnz: Maximum number of columns to select, i.e. the class's coreset budget.
    :param lam: Tikhonov regularization coefficient for the least-squares problem at each step.
    :param positive: If True, enforce non-negative weights by gradually dropping columns with negative weights.
    :param tol: Early-stopping threshold on the ratio of the residual norm to the target norm.
    :param desc: Label shown on the progress bar.

    :return: Tuple (weights, positions) — the list of weights and the list of selected column positions, aligned in order, where a position is an index in [0, m).
    """
    d, m = A.shape
    device, dtype = A.device, A.dtype
    normb = b.norm().item()
    if normb == 0.0:
        normb = 1.0
    resid = b.clone()
    positions = []
    x_sel = None

    for _ in tqdm(range(nnz), desc=desc):
        if resid.norm().item() / normb < tol:
            break
        proj = A.t() @ resid
        scores = proj.clone() if positive else proj.abs()
        if positions:
            scores[torch.tensor(positions, dtype=torch.long, device=device)] = float("-inf")
        j = int(torch.argmax(scores).item())
        if j in positions:
            break
        positions.append(j)

        A_sel = A[:, positions]
        k = len(positions)
        G = A_sel.t() @ A_sel + lam * torch.eye(k, device=device, dtype=dtype)
        x_sel = torch.linalg.solve(G, A_sel.t() @ b)

        if positive:
            while x_sel.numel() > 0 and x_sel.min().item() < 0.0:
                neg = int(torch.argmin(x_sel).item())
                positions.pop(neg)
                if not positions:
                    break
                A_sel = A[:, positions]
                k = len(positions)
                G = A_sel.t() @ A_sel + lam * torch.eye(k, device=device, dtype=dtype)
                x_sel = torch.linalg.solve(G, A_sel.t() @ b)

        if not positions:
            break
        resid = b - A_sel @ x_sel

    if x_sel is None or not positions:
        return [], []
    return x_sel.detach().cpu().tolist(), list(positions)


def gradmatch_omp(
    gradients,
    budget,
    target=None,
    candidate_indices_local=None,
    desc="GradMatch",
    lam=1.0,
    positive=True,
    tol=1e-4,
    normalize_weights=True,
    force_cpu=True,
):
    """
    Select a coreset on a precomputed gradient matrix via OMP, returning local indices and weights.
    :param gradients: Tensor (n, feature_dim) of gradients of all samples.
    :param budget: Number of samples to select.
    :param target: Target vector (feature_dim,) to fit. Default None means use the sum of the candidates' gradients, matching the per-class GradMatchPB variant.
    :param candidate_indices_local: List of local indices allowed to be selected. None uses all.
    :param desc: Label shown on the progress bar.
    :param lam: Regularization coefficient for OMP.
    :param positive: Enforce non-negative weights (default True, as in standard GradMatch).
    :param tol: Early-stopping threshold on the residual-norm ratio.
    :param normalize_weights: If True (default), normalize the weights to mean 1.0. Set False if the trainer normalizes by the sum of the weights itself.
    :param force_cpu: True (default) moves gradients to CPU before running; False keeps them on the current device.
    
    :return: Tuple (selected_local, weights_local) — the list of selected local indices and a dict local index -> weight. If OMP returns fewer than budget, the shortfall is padded with
    remaining candidates chosen at random with weight 1.0 to keep the exact coreset size.
    """
    if force_cpu:
        gradients = gradients.detach().cpu().float()
    else:
        gradients = gradients.detach().float()
    print(f"[GradMatch] gradients.device = {gradients.device}")
    print(f"[GradMatch] gradients.shape  = {gradients.shape}")

    n = gradients.shape[0]
    if candidate_indices_local is None:
        candidate_indices_local = list(range(n))
    candidate_indices_local = list(dict.fromkeys(candidate_indices_local))
    m = len(candidate_indices_local)

    if budget >= m:
        chosen = candidate_indices_local[:budget]
        return chosen, {idx: 1.0 for idx in chosen}

    device = gradients.device
    cand = torch.tensor(candidate_indices_local, dtype=torch.long, device=device)
    A = gradients[cand].t().contiguous()            # (d, m), each column is one candidate
    b = A.sum(dim=1) if target is None else target.to(device).float()

    weights, positions = _omp_reg(
        A=A, b=b, nnz=min(budget, m),
        lam=lam, positive=positive, tol=tol, desc=desc,
    )

    selected_local = [candidate_indices_local[p] for p in positions]
    weights_local = {
        candidate_indices_local[p]: float(w) for p, w in zip(positions, weights)
    }

    # Normalize to mean 1.0 so the weights are on the same scale as CRAIG, avoiding an inflated effective LR.
    if normalize_weights and weights_local:
        mean_w = sum(weights_local.values()) / len(weights_local)
        if mean_w > 0:
            weights_local = {k: v / mean_w for k, v in weights_local.items()}

    # Pad up to budget if OMP stops early, keeping the coreset size the pipeline expects.
    if len(selected_local) < budget:
        chosen_set = set(selected_local)
        remaining = [idx for idx in candidate_indices_local if idx not in chosen_set]
        need = budget - len(selected_local)
        if remaining:
            perm = torch.randperm(len(remaining))[:need].tolist()
            for p in perm:
                idx = remaining[p]
                selected_local.append(idx)
                weights_local[idx] = 1.0

    return selected_local, weights_local


def select_gradmatch_coreset(
    # Mode 1: GradMatch for the full dataset
    model,
    full_dataset,
    indices_by_class,
    coreset_size_fraction,
    device,
    gradient_type,
    num_classes=10,
    gradient_batch_size=128,

    # Mode 2: GradMatch on a precomputed gradient matrix
    gradients=None,
    budget=None,
    candidate_indices_local=None,
    target=None,
    desc="GradMatch",

    # OMP hyperparameters
    lam=1.0,
    positive=True,
    tol=1e-4,
    normalize_weights=True,
    force_cpu=True,
):
    """
    Orchestrates GradMatch coreset selection, supporting two modes like CRAIG:
    (1) run over the full full_dataset per class, computing the gradients and per-class budget itself;
    (2) run directly on a precomputed gradients/candidate_indices_local matrix.
    :param model: Model used to compute the gradient (mode 1).
    :param full_dataset: The full original dataset (mode 1).
    :param indices_by_class: Dict class label -> list of indices in full_dataset (mode 1).
    :param coreset_size_fraction: Coreset size as a fraction of full_dataset (mode 1).
    :param device: Device used when computing gradients (mode 1).
    :param gradient_type: Type of gradient representation to compute (mode 1); should match CRAIG for a fair comparison.
    :param num_classes: Total number of classes (mode 1).
    :param gradient_batch_size: Batch size when computing gradients (mode 1).
    :param gradients: Precomputed gradient matrix; if not None, switches to mode 2.
    :param budget: Number of samples to select (mode 2).
    :param candidate_indices_local: List of local candidate indices (mode 2).
    :param target: Target vector to fit (mode 2); None means the sum of the candidates' gradients.
    :param desc: Label shown on the progress bar (mode 2).
    :param lam: OMP regularization coefficient.
    :param positive: Enforce non-negative weights.
    :param tol: OMP early-stopping threshold.
    :param force_cpu: Move gradients to CPU before running OMP.
    
    :return:
        - Mode 2: tuple (selected_local, weights_local).
        - Mode 1: triple (final_indices, final_weights, elapsed_time).
    """
    # ==========================================================
    # Mode 2: GradMatch on a precomputed gradient/candidate matrix
    # ==========================================================
    if gradients is not None:
        return gradmatch_omp(
            gradients=gradients,
            budget=budget,
            target=target,
            candidate_indices_local=candidate_indices_local,
            desc=desc,
            lam=lam,
            positive=positive,
            tol=tol,
            normalize_weights=normalize_weights,
            force_cpu=force_cpu,
        )

    # ==========================================================
    # Mode 1: GradMatch for the full dataset, run per class
    # ==========================================================
    print(">>> Selecting coreset by GradMatch...")
    start_time = time.time()
    final_indices, final_weights = [], {}
    total_size = max(1, int(len(full_dataset) * coreset_size_fraction))
    class_sizes = {
        c: len(indices_by_class[c])
        for c in indices_by_class
        if len(indices_by_class[c]) > 0
    }
    total_n = sum(class_sizes.values())
    allocated = {c: int(total_size * class_sizes[c] / total_n) for c in class_sizes}
    remaining = total_size - sum(allocated.values())
    for c in sorted(allocated, key=lambda x: -((total_size * class_sizes[x] / total_n) - allocated[x])):
        if remaining <= 0:
            break
        allocated[c] += 1
        remaining -= 1

    for c, class_budget in allocated.items():
        class_indices = indices_by_class[c]
        if class_budget <= 0 or len(class_indices) == 0:
            continue
        print(
            f"\n>>> GradMatch for class {c}: "
            f"budget={class_budget}, samples={len(class_indices)}"
        )

        class_gradients = compute_gradient_representations(
            model=model,
            full_dataset=full_dataset,
            indices=class_indices,
            device=device,
            gradient_type=gradient_type,
            batch_size=gradient_batch_size,
            return_device="device",
        )

        if class_gradients.shape[0] == 0:
            continue

        selected_local, weights_local = gradmatch_omp(
            gradients=class_gradients,
            budget=class_budget,
            target=None,
            candidate_indices_local=None,
            desc=f"GradMatch for class {c}",
            lam=lam,
            positive=positive,
            tol=tol,
            normalize_weights=normalize_weights,
            force_cpu=force_cpu,
        )

        for local_idx, weight in weights_local.items():
            global_idx = class_indices[local_idx]
            final_indices.append(global_idx)
            final_weights[global_idx] = float(weight)

    end_time = time.time()
    assert len(final_indices) == total_size, (
        f"GradMatch size mismatch: expected {total_size}, got {len(final_indices)}")
    print(f"\n>>> GradMatch selection done in {end_time - start_time:.2f}s")
    print(f">>> Coreset size: {len(final_indices)}")
    return final_indices, final_weights, end_time - start_time