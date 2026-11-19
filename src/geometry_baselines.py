"""
---file geometry_baselines.py---
Geometry-based coreset selection methods, common in the DeepCore benchmark, include: Herding and k-Center Greedy.
"""
import time
import numpy as np
import torch
from tqdm import tqdm

from utils import compute_gradient_representations, calculate_weights


# ==============================================================================
# HERDING (Welling 2009)
# ==============================================================================

def herding_select(X: np.ndarray, budget: int) -> list:
    """
    Herding algorithm: iteratively selects points so that the running mean of the selected set gets as close as possible to the mean of all of X. 
    
    :param X: Data (gradient) matrix of one class, shape (n, d).
    :param budget: Number of points to select.
    :return: List of local indices (within X) that were selected, in selection order.
    """
    n = X.shape[0]
    budget = min(budget, n)
    mu = X.mean(axis=0)
    selected = []
    running_sum = np.zeros(X.shape[1])
    remaining = list(range(n))
    for k in range(1, budget + 1):
        cand_X = X[remaining]
        # Distance to center mu if each candidate were added to the selected set
        candidate_means = (running_sum + cand_X) / k
        dists = np.linalg.norm(mu - candidate_means, axis=1)
        best_pos = int(np.argmin(dists))
        best_idx = remaining[best_pos]
        selected.append(best_idx)
        running_sum += X[best_idx]
        remaining.pop(best_pos)
    return selected


def select_herding_coreset(
        model,
        full_dataset,
        indices_by_class,
        coreset_size_fraction,
        device,
        gradient_type,
        num_classes,
        gradient_batch_size,
        log_selected_indices=False,
    ):
    """
    Orchestrates Herding coreset selection on the gradient matrix, per class.
    
    :param model: Model used to compute the gradient for each sample.
    :param full_dataset: The full original dataset.
    :param indices_by_class: Dict mapping a class label to its list of indices in full_dataset.
    :param coreset_size_fraction: Coreset size as a fraction of full_dataset.
    :param device: Device (CPU/GPU) used when computing gradients.
    :param gradient_type: Type of gradient representation to compute.
    :param num_classes: Total number of classes in the dataset.
    :param gradient_batch_size: Batch size when computing gradients.
    :param log_selected_indices: If True, log the selected local/global indices.
    :return: Tuple (final_indices, final_weights, elapsed_time) — global indices of the coreset, their corresponding weights, and the elapsed time (seconds).
    """
    print(">>> Selecting coreset by Herding...")
    start_time = time.time()
    final_indices, final_weights = [], {}
    total_size = max(1, int(len(full_dataset) * coreset_size_fraction))
    class_sizes = {c: len(indices_by_class[c]) for c in indices_by_class if len(indices_by_class[c]) > 0}
    total_n = sum(class_sizes.values())
    allocated = {c: int(total_size * class_sizes[c] / total_n) for c in class_sizes}
    remaining = total_size - sum(allocated.values())
    for c in sorted(allocated, key=lambda x: -((total_size * class_sizes[x] / total_n) - allocated[x])):
        if remaining <= 0:
            break
        allocated[c] += 1
        remaining -= 1

    for c, budget in allocated.items():
        class_indices = indices_by_class[c]
        if budget <= 0 or len(class_indices) == 0:
            continue
        print(f"\n>>> Herding for class {c}: budget={budget}, samples={len(class_indices)}")

        class_gradients = compute_gradient_representations(
            model=model, full_dataset=full_dataset, indices=class_indices,
            device=device, gradient_type=gradient_type, batch_size=gradient_batch_size)

        X = class_gradients.detach().cpu().numpy().astype(np.float64)
        n_c = X.shape[0]

        if budget >= n_c:
            selected_local = list(range(n_c))
            print(f"Class {c}: budget >= samples, get all {n_c} sample / budget={budget}")
        else:
            selected_local = herding_select(X, budget)
            print(f"Class {c}: Herding selected {len(selected_local)} points / budget={budget}")

        selected_global = [class_indices[i] for i in selected_local]
        if log_selected_indices:
            print(f"\n[Herding selected indices] class={c}")
            print(f"local_indices : {selected_local}")
            print(f"global_indices: {selected_global}")

        weights_local = calculate_weights(
            class_gradients=class_gradients,
            selected_local=selected_local,
        )

        for local_idx, weight in weights_local.items():
            global_idx = class_indices[local_idx]
            final_indices.append(global_idx)
            final_weights[global_idx] = float(weight)

    end_time = time.time()
    print(f"\n>>> Herding selection done in {end_time - start_time:.2f}s")
    print(f">>> Coreset size: {len(final_indices)}")

    if log_selected_indices:
        print("\n[Herding final coreset global indices]")
        print(final_indices)

    return final_indices, final_weights, end_time - start_time


# ==============================================================================
# K-CENTER GREEDY (Sener & Savarese, ICLR 2018)
# ==============================================================================

def kcenter_greedy_cdist(
        gradients,
        budget,
        candidate_indices_local=None,
        desc="kcenter",
        force_cpu=True,
        seed=None,
    ):
    """
    k-Center Greedy algorithm: iterate `budget` times, each time selecting the point farthest from the selected set — minimizes the largest covering radius.
    
    :param gradients: Gradient tensor of all samples, shape (n, feature_dim).
    :param budget: Number of samples to select into the coreset.
    :param candidate_indices_local: List of local indices allowed as candidates. Default None means use all n samples.
    :param desc: Label shown on the progress bar (tqdm).
    :param force_cpu: If True (default), always move gradients to CPU before running. If False, keep gradients on the current device.
    :param seed: Seed for choosing the first initialization point (random among candidates).
    :return: List of local indices (within gradients) of the selected samples. If budget >= number of candidates, returns all candidates (truncated to budget).
    """
    if force_cpu:
        gradients = gradients.detach().cpu().float()
    else:
        gradients = gradients.detach().float()

    n = gradients.shape[0]
    if candidate_indices_local is None:
        candidate_indices_local = list(range(n))
    candidate_indices_local = list(dict.fromkeys(candidate_indices_local))
    if budget >= len(candidate_indices_local):
        return candidate_indices_local[:budget]

    rng = np.random.default_rng(seed)
    # Initialization point: pick one random candidate (there is no natural "center"
    # as in CRAIG, which starts with min_dists=inf; k-center needs a first anchor point)
    first_idx = int(rng.choice(candidate_indices_local))
    selected_local = [first_idx]
    selected_set = {first_idx}

    min_dists = torch.cdist(
        gradients, gradients[first_idx].view(1, -1), p=2
    ).squeeze(1) ** 2

    for _ in tqdm(range(budget - 1), desc=desc):
        remaining = [idx for idx in candidate_indices_local if idx not in selected_set]
        if len(remaining) == 0:
            break
        remaining_tensor = torch.tensor(remaining, dtype=torch.long, device=gradients.device)
        remaining_min_dists = min_dists[remaining_tensor]
        best_pos = int(torch.argmax(remaining_min_dists).item())
        best_idx = remaining[best_pos]
        selected_local.append(best_idx)
        selected_set.add(best_idx)

        dist_to_new = torch.cdist(
            gradients, gradients[best_idx].view(1, -1), p=2
        ).squeeze(1) ** 2
        min_dists = torch.minimum(min_dists, dist_to_new)

    return selected_local


def select_kcenter_coreset(
        model,
        full_dataset,
        indices_by_class,
        coreset_size_fraction,
        device,
        gradient_type,
        num_classes,
        gradient_batch_size,
        random_state=None,
        force_cpu=True,
        log_selected_indices=False,
    ):
    """
    Orchestrates k-Center Greedy coreset selection on the gradient matrix, per class — same structure as select_craig_coreset (craig.py).
    
    :param model: Model used to compute the gradient for each sample.
    :param full_dataset: The full original dataset.
    :param indices_by_class: Dict mapping a class label to its list of indices in full_dataset.
    :param coreset_size_fraction: Coreset size as a fraction of full_dataset.
    :param device: Device (CPU/GPU) used when computing gradients.
    :param gradient_type: Type of gradient representation to compute.
    :param num_classes: Total number of classes in the dataset.
    :param gradient_batch_size: Batch size when computing gradients.
    :param random_state: Seed for choosing the first initialization point of each class.
    :param force_cpu: Passed straight to kcenter_greedy_cdist — True (default) forces CPU, False allows running on GPU (see craig_greedy_cdist for the trade-off).
    :param log_selected_indices: If True, log the selected local/global indices.
    :return: Tuple (final_indices, final_weights, elapsed_time) — global indices of the coreset, their corresponding weights, and the elapsed time (seconds).
    """
    print(">>> Selecting coreset by k-Center Greedy...")
    start_time = time.time()
    final_indices, final_weights = [], {}
    total_size = max(1, int(len(full_dataset) * coreset_size_fraction))
    class_sizes = {c: len(indices_by_class[c]) for c in indices_by_class if len(indices_by_class[c]) > 0}
    total_n = sum(class_sizes.values())
    allocated = {c: int(total_size * class_sizes[c] / total_n) for c in class_sizes}
    remaining = total_size - sum(allocated.values())
    for c in sorted(allocated, key=lambda x: -((total_size * class_sizes[x] / total_n) - allocated[x])):
        if remaining <= 0:
            break
        allocated[c] += 1
        remaining -= 1

    for c, budget in allocated.items():
        class_indices = indices_by_class[c]
        if budget <= 0 or len(class_indices) == 0:
            continue
        print(f"\n>>> k-Center Greedy for class {c}: budget={budget}, samples={len(class_indices)}")

        class_gradients = compute_gradient_representations(
            model=model, full_dataset=full_dataset, indices=class_indices,
            device=device, gradient_type=gradient_type, batch_size=gradient_batch_size,
            return_device="device")

        n_c = class_gradients.shape[0]
        if n_c == 0:
            continue

        if budget >= n_c:
            selected_local = list(range(n_c))
            print(f"Class {c}: budget >= samples, get all {n_c} samples / budget={budget}")
        else:
            selected_local = kcenter_greedy_cdist(
                gradients=class_gradients, budget=budget,
                desc=f"kCenter for class {c}",
                force_cpu=force_cpu, seed=random_state,
            )
            print(f"Class {c}: k-Center Greedy selected {len(selected_local)} points / budget={budget}")

        selected_global = [class_indices[i] for i in selected_local]
        if log_selected_indices:
            print(f"\n[k-Center Greedy selected indices] class={c}")
            print(f"local_indices : {selected_local}")
            print(f"global_indices: {selected_global}")

        weights_local = calculate_weights(
            class_gradients=class_gradients,
            selected_local=selected_local,
        )

        for local_idx, weight in weights_local.items():
            global_idx = class_indices[local_idx]
            final_indices.append(global_idx)
            final_weights[global_idx] = float(weight)

    end_time = time.time()
    print(f"\n>>> k-Center Greedy selection done in {end_time - start_time:.2f}s")
    print(f">>> Coreset size: {len(final_indices)}")

    if log_selected_indices:
        print("\n[k-Center Greedy final coreset global indices]")
        print(final_indices)

    return final_indices, final_weights, end_time - start_time
