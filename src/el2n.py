"""
---file el2n.py---
Coreset selection based on the EL2N / GraNd score (Paul et al., "Deep Learning on a Data Diet", NeurIPS 2021)
"""
import time
import torch

from utils import compute_gradient_representations, calculate_weights


def el2n_score_select(gradients, budget, force_cpu=True):
    """
    Compute the EL2N score (= norm(gradient)) for each sample, then select the budget
    samples with the highest scores.

    :param gradients: Gradient tensor of all samples in the class, shape (n, feature_dim).
    :param budget: Number of samples to select.
    :param force_cpu:  Passed straight to el2n_score_select — True (default) forces CPU (safe, deterministic), False allows running on GPU (faster, no bit-for-bit reproducibility guarantee versus CPU).
    
    :return: List of local indices (within gradients) of the selected samples, ordered by descending score. If budget >= n, returns all indices (sorted by score).
    """
    if force_cpu:
        g = gradients.detach().cpu().float()
    else:
        g = gradients.detach().float()

    n = g.shape[0]
    budget = min(budget, n)

    scores = torch.norm(g, dim=1)
    order = torch.argsort(scores, descending=True)
    return order[:budget].tolist()


def select_el2n_coreset(
        model,
        full_dataset,
        indices_by_class,
        coreset_size_fraction,
        device,
        gradient_type,
        num_classes,
        gradient_batch_size,
        force_cpu=True,
        log_selected_indices=False,
    ):
    """
    Orchestrates EL2N/GraNd score-based coreset selection on the gradient matrix, per class.

    :param model: Model used to compute the gradient for each sample.
    :param full_dataset: The full original dataset.
    :param indices_by_class: Dict mapping a class label to its list of indices in full_dataset.
    :param coreset_size_fraction: Coreset size as a fraction of full_dataset.
    :param device: Device (CPU/GPU) used when computing gradients.
    :param gradient_type: Type of gradient representation to compute (recommended "logit" so the score matches the original EL2N definition).
    :param num_classes: Total number of classes in the dataset.
    :param gradient_batch_size: Batch size when computing gradients.
    :param force_cpu: Passed straight to el2n_score_select — True (default) forces CPU (safe, deterministic), False allows running on GPU (faster, no bit-for-bit reproducibility guarantee versus CPU).
    :param log_selected_indices: If True, log the selected local/global indices.
    
    :return: Tuple (final_indices, final_weights, elapsed_time) — global indices of the coreset, their corresponding weights, and the elapsed time (seconds).
    """
    print(f">>> Selecting coreset by EL2N/GraNd (score-based, force_cpu={force_cpu})...")
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
        print(f"\n>>> EL2N for class {c}: budget={budget}, samples={len(class_indices)}")

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
            selected_local = el2n_score_select(
                gradients=class_gradients, budget=budget, force_cpu=force_cpu)
            print(f"Class {c}: EL2N selected {len(selected_local)} points / budget={budget}")

        selected_global = [class_indices[i] for i in selected_local]
        if log_selected_indices:
            print(f"\n[EL2N selected indices] class={c}")
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
    print(f"\n>>> EL2N selection done in {end_time - start_time:.2f}s")
    print(f">>> Coreset size: {len(final_indices)}")

    if log_selected_indices:
        print("\n[EL2N final coreset global indices]")
        print(final_indices)

    return final_indices, final_weights, end_time - start_time