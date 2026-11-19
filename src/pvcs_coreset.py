"""
pvcs_coreset.py  

Main script for algorithm Pairwise Vector-based Coreset Selection (PVCS) in manuscript:
`Stable and Data-Efficient Coreset Selection through Pairwise Class Geometry` (Vu Hoai Thu, Nguyen Kieu Linh)
"""

import os
import csv
import time
import numpy as np
from numpy.linalg import norm
import warnings

from utils import compute_gradient_representations, calculate_weights


# ==============================================================================
# ALGO BVS: inter-centroid direction + take boundary and center 
# ==============================================================================

def BVS(X: np.ndarray, labels: np.ndarray,
        target_ratio: float,
        edge_ratio: float = 0.5,
        return_stats: bool = False,
        middle_mode: str = "origin"):
    """
    middle_mode: which criterion defines the "middle".
      
    """
    """
    Bipartite Vector Selection (Algorithm 1): select a coreset from a two-class pair by projecting the logit-gradients 
    onto the inter-centroid direction and taking the extreme (hardest) and core (most confident) samples.

    :param X: Logit-gradient embeddings of the pair, shape (n, d).
    :param labels: Binary labels of the pair, shape (n,); the two classes are taken in sorted order as 0 and 1.
    :param target_ratio: Fraction tau of the pair to select; the budget is K = ceil(tau * n).
    :param edge_ratio: r in [0, 1], the share of the budget spent on extreme samples (K_e = max(1, floor(K * r))); the rest goes to core samples (K_m = K - K_e).
    :param return_stats: If True, also return the sets of extreme and core indices.
    :param middle_mode: Criterion for the "middle"/core samples.
        "origin"   (default): |projection| ≈ 0  -> near the origin (in gradient space = easy/confident points).
        "midpoint" : |projection - m| ≈ 0 with m = the projection of the midpoint of the two centroids -> points near the separating hyperplane between the two classes (the true boundary).
    
    :return: A sorted array of the selected local indices (within X).
    """
    n = X.shape[0]
    target_count = max(1, int(np.ceil(target_ratio * n)))

    classes = np.unique(labels)
    if len(classes) < 2:
        warnings.warn("Algo BVS: Only 1 class.")
        norms = np.linalg.norm(X, axis=1)
        result = np.argsort(-norms)[:target_count]
        if return_stats:
            return result, set(), set()
        return result

    c0 = X[labels == classes[0]].mean(axis=0)
    c1 = X[labels == classes[1]].mean(axis=0)
    direction = c0 - c1
    dir_norm = norm(direction)
    if dir_norm < 1e-9:
        direction = np.random.randn(X.shape[1])
        direction /= norm(direction)
    else:
        direction /= dir_norm

    projections = X @ direction
    # threshold for "middle": 0 if origin, projected-midpoint-of-the-two-centroids if midpoint
    middle_center = 0.0 if middle_mode == "origin" else float(0.5 * (c0 + c1) @ direction)

    edge_budget = max(1, int(np.floor(target_count * edge_ratio)))
    middle_budget = target_count - edge_budget

    V_indices = set()
    edge_indices = set()

    sorted_desc = np.argsort(-projections)
    sorted_asc = np.argsort(projections)
    left, right = 0, 0

    while len(V_indices) < edge_budget:
        if right < n and len(V_indices) < edge_budget:
            idx = int(sorted_desc[right])
            V_indices.add(idx)
            edge_indices.add(idx)
            right += 1
        if left < n and len(V_indices) < edge_budget:
            idx = int(sorted_asc[left])
            V_indices.add(idx)
            edge_indices.add(idx)
            left += 1
        if left >= n and right >= n:
            break

    middle_indices = set()
    if middle_budget > 0:
        abs_proj = np.abs(projections - middle_center)   # origin: |proj|; midpoint: |proj - m|
        sorted_by_abs = np.argsort(abs_proj)
        added = 0
        for idx in sorted_by_abs:
            if idx not in V_indices:
                idx = int(idx)
                V_indices.add(idx)
                middle_indices.add(idx)
                added += 1
                if added >= middle_budget:
                    break

    result = np.array(sorted(V_indices))
    if len(result) > target_count:
        trimmed_out = set(result[target_count:].tolist())
        result = result[:target_count]
        edge_indices -= trimmed_out
        middle_indices -= trimmed_out

    if return_stats:
        return result, edge_indices, middle_indices
    return result


# ==============================================================================
# Utilities: nearest class pairs + per-class neighbors
# ==============================================================================

def _build_nearest_class_pairs(centroids: dict, max_neighbor_classes: int) -> set:
    classes = list(centroids.keys())
    pairs = set()
    for c in classes:
        dists = {j: norm(centroids[c] - centroids[j]) for j in classes if j != c}
        nearest = sorted(dists, key=dists.get)[:max_neighbor_classes]
        for j in nearest:
            pairs.add((c, j) if c < j else (j, c))
    return pairs


def _nearest_neighbor_classes(centroids: dict, c, max_neighbor_classes: int) -> list:
    """Return the list of the M classes nearest to c by centroid distance (excluding c)."""
    others = [j for j in centroids if j != c]
    others.sort(key=lambda j: norm(centroids[c] - centroids[j]))
    return others[:max_neighbor_classes]


# ==============================================================================
# PVCS: per-class top-k, 100% geometric (For Ablation)
# ==============================================================================

def PVCS_perclass(X: np.ndarray, labels: np.ndarray,
                  target_ratio: float,
                  edge_ratio: float = 0.5,
                  max_neighbor_classes: int = 2,
                  random_state: int | None = None,
                  return_stats: bool = False) -> np.ndarray:
    n = X.shape[0]
    classes = np.unique(labels)
    indices_by_class = {c: np.where(labels == c)[0] for c in classes}

    if len(classes) < 2:
        warnings.warn("PVCS: Only 1 class, fallback norm.")
        target_count = max(1, int(np.ceil(target_ratio * n)))
        norms = np.linalg.norm(X, axis=1)
        result = np.sort(np.argsort(-norms)[:target_count])
        if return_stats:
            return result, {}
        return result

    # Step 1: per-class budget allocation (largest-remainder) -- UNCHANGED
    total_target = max(1, int(np.ceil(target_ratio * n)))
    class_sizes = {c: len(indices_by_class[c]) for c in classes}
    total_n = sum(class_sizes.values())
    allocated = {c: int(total_target * class_sizes[c] / total_n) for c in classes}
    remaining = total_target - sum(allocated.values())
    for c in sorted(allocated, key=lambda x: -((total_target * class_sizes[x] / total_n) - allocated[x])):
        if remaining <= 0:
            break
        allocated[c] += 1
        remaining -= 1

    # Step 2: class centroids + nearest neighbors per EACH class
    centroids = {c: X[indices_by_class[c]].mean(axis=0) for c in classes}
    max_neighbors = min(max_neighbor_classes, len(classes) - 1)

    final_indices = []
    edge_by_class = {c: set() for c in classes}
    middle_by_class = {c: set() for c in classes}

    # Steps 3+4: per-point geometric scoring, then per-class top-k selection
    for c in classes:
        budget_c = allocated[c]
        cls_idx = indices_by_class[c]           # global indices of class-c points
        n_c = len(cls_idx)
        if budget_c <= 0 or n_c == 0:
            continue
        budget_c = min(budget_c, n_c)           # safety (always holds when fraction<1)

        Xc = X[cls_idx]                         # (n_c, d)
        neighbors = _nearest_neighbor_classes(centroids, c, max_neighbors)

        # for each inter-centroid direction d_{c,j}: signed = x·d - m ; m = midpoint of the two centroids
        signed = np.full((n_c, max(1, len(neighbors))), -np.inf)
        absdist = np.full((n_c, max(1, len(neighbors))), np.inf)
        valid = False
        for k, j in enumerate(neighbors):
            diff = centroids[c] - centroids[j]
            dn = norm(diff)
            if dn < 1e-9:
                continue
            d = diff / dn
            m = float(0.5 * ((centroids[c] + centroids[j]) @ d))   # boundary threshold
            proj = Xc @ d - m
            signed[:, k] = proj
            absdist[:, k] = np.abs(proj)
            valid = True

        if not valid:
            # rare fallback: centroids coincide -> use norm as the representative score
            edge_score = np.linalg.norm(Xc, axis=1)
            boundary_score = -edge_score
        else:
            edge_score = signed.max(axis=1)      # deepest toward its own class (representative)
            boundary_score = absdist.min(axis=1)  # closest to the nearest boundary (boundary)

        edge_budget = int(np.floor(budget_c * edge_ratio))
        edge_budget = max(0, min(edge_budget, n_c))
        middle_budget = budget_c - edge_budget

        # edge: top by descending edge_score
        order_edge = np.argsort(-edge_score)
        edge_local = [int(i) for i in order_edge[:edge_budget]]
        edge_set_local = set(edge_local)

        # middle: from the REMAINING points, closest to the boundary (smallest boundary_score)
        middle_local = []
        if middle_budget > 0:
            order_mid = np.argsort(boundary_score)
            for i in order_mid:
                i = int(i)
                if i in edge_set_local:
                    continue
                middle_local.append(i)
                if len(middle_local) >= middle_budget:
                    break

        chosen_local = edge_local + middle_local

        # fallback: if rounding leaves a shortfall, take more by edge_score (still 100% geometric)
        if len(chosen_local) < budget_c:
            chosen_set = set(chosen_local)
            for i in order_edge:
                i = int(i)
                if i in chosen_set:
                    continue
                chosen_local.append(i)
                chosen_set.add(i)
                if len(chosen_local) >= budget_c:
                    break

        edge_by_class[c] = set(int(cls_idx[i]) for i in edge_local)
        middle_by_class[c] = set(int(cls_idx[i]) for i in middle_local)
        for i in chosen_local:
            final_indices.append(int(cls_idx[i]))

    result = np.array(sorted(final_indices))

    if return_stats:
        stats_by_class = {
            c: {
                'edge': len(edge_by_class[c]),
                'middle': len(middle_by_class[c]),
                'random_filled': 0,   # the new PVCS does no random fill
                'trimmed': 0,         # and no trim either
            }
            for c in classes
        }
        return result, stats_by_class
    return result


# ==============================================================================
# PVCS: pairwise multi-class extension (Algorithm 2)
# ==============================================================================

def PVCS_pairwise(X: np.ndarray, labels: np.ndarray,
                  target_ratio: float,
                  edge_ratio: float = 0.5,
                  max_neighbor_classes: int = 2,
                  random_state: int | None = None,
                  return_stats: bool = False,
                  middle_mode: str = "origin") -> np.ndarray:

    """
    PVCS multi-class extension (Algorithm 2): the pairwise wrapper around BVS with a random backbone (fill_mode="random").

    :param X: Logit-gradient embeddings of all samples, shape (n, d).
    :param labels: Class labels of all samples, shape (n,).
    :param target_ratio: Total coreset fraction tau; the budget is K = ceil(tau * n).
    :param edge_ratio: r in [0, 1], the extreme-vs-core split passed to BVS.
    :param max_neighbor_classes: M, the number of nearest-neighbour classes (by centroid distance) paired with each class.
    :param random_state: Seed for the uniform random fill that completes each class budget.
    :param return_stats: If True, also return a per-class statistics dict.
    :param middle_mode: Criterion for the core samples, passed to BVS ("origin" or "midpoint").
    
    :return: A sorted array of the selected global indices (within X).
    """
    n = X.shape[0]
    classes = np.unique(labels)
    indices_by_class = {c: np.where(labels == c)[0] for c in classes}

    if len(classes) < 2:
        warnings.warn("PVCS: Only 1 class, fallback norm.")
        target_count = max(1, int(np.ceil(target_ratio * n)))
        norms = np.linalg.norm(X, axis=1)
        result = np.sort(np.argsort(-norms)[:target_count])
        if return_stats:
            return result, {}
        return result

    # Step 1: per-class budget allocation
    total_target = max(1, int(np.ceil(target_ratio * n)))
    class_sizes = {c: len(indices_by_class[c]) for c in classes}
    total_n = sum(class_sizes.values())
    allocated = {c: int(total_target * class_sizes[c] / total_n) for c in classes}
    remaining = total_target - sum(allocated.values())
    for c in sorted(allocated, key=lambda x: -((total_target * class_sizes[x] / total_n) - allocated[x])):
        if remaining <= 0:
            break
        allocated[c] += 1
        remaining -= 1

    # Step 2: build the set of class pairs
    centroids = {c: X[indices_by_class[c]].mean(axis=0) for c in classes}
    max_neighbors = min(max_neighbor_classes, len(classes) - 1)
    class_pairs = _build_nearest_class_pairs(centroids, max_neighbors)

    pairs_per_class = {c: 0 for c in classes}
    for (i, j) in class_pairs:
        pairs_per_class[i] += 1
        pairs_per_class[j] += 1

    # Step 3: run BVS on each pair
    selected_by_class = {c: set() for c in classes}
    edge_by_class = {c: set() for c in classes}
    middle_by_class = {c: set() for c in classes}
    for (i, j) in class_pairs:
        pair_idx = np.concatenate([indices_by_class[i], indices_by_class[j]])
        X_pair = X[pair_idx]
        labels_pair = labels[pair_idx]
        pair_n = len(pair_idx)

        share_i = allocated[i] / pairs_per_class[i] if pairs_per_class[i] > 0 else 0
        share_j = allocated[j] / pairs_per_class[j] if pairs_per_class[j] > 0 else 0
        pair_target_count = max(1, int(np.ceil(share_i + share_j)))
        pair_ratio = min(1.0, pair_target_count / pair_n)

        sel_local, edge_local, middle_local = BVS(
            X_pair, labels_pair,
            target_ratio=pair_ratio,
            edge_ratio=edge_ratio,
            return_stats=True,
            middle_mode=middle_mode)
        sel_global = pair_idx[sel_local]

        for g_idx in sel_global:
            selected_by_class[labels[g_idx]].add(int(g_idx))

        for local_idx in edge_local:
            g_idx = int(pair_idx[local_idx])
            c = labels[g_idx]
            if g_idx not in middle_by_class[c]:
                edge_by_class[c].add(g_idx)
        for local_idx in middle_local:
            g_idx = int(pair_idx[local_idx])
            c = labels[g_idx]
            if g_idx not in edge_by_class[c]:
                middle_by_class[c].add(g_idx)

    # Step 4: fill shortfalls / trim surplus
    rng = np.random.default_rng(random_state)
    final_indices = []
    random_filled_by_class = {c: 0 for c in classes}
    trimmed_by_class = {c: 0 for c in classes}
    for c in classes:
        budget_c = allocated[c]
        if budget_c <= 0:
            continue
        sel = list(selected_by_class[c])
        if len(sel) < budget_c:
            selected_set = set(sel)
            remaining_candidates = [i for i in indices_by_class[c] if i not in selected_set]
            rng.shuffle(remaining_candidates)
            need = budget_c - len(sel)
            chosen = remaining_candidates[:need]
            sel.extend(chosen)
            random_filled_by_class[c] = len(chosen)
        if len(sel) > budget_c:
            trimmed_out = set(sel[budget_c:])
            trimmed_by_class[c] = len(trimmed_out)
            sel = sel[:budget_c]
            edge_by_class[c] -= trimmed_out
            middle_by_class[c] -= trimmed_out
        final_indices.extend(sel)

    result = np.array(sorted(final_indices))

    if return_stats:
        stats_by_class = {
            c: {
                'edge': len(edge_by_class[c]),
                'middle': len(middle_by_class[c]),
                'random_filled': random_filled_by_class[c],
                'trimmed': trimmed_by_class[c],
            }
            for c in classes
        }
        return result, stats_by_class
    return result


# ==============================================================================
# DISPATCH: PVCS(...) selects the algorithm by fill_mode
# ==============================================================================

def PVCS(X: np.ndarray, labels: np.ndarray,
         target_ratio: float,
         edge_ratio: float = 0.5,
         max_neighbor_classes: int = 2,
         random_state: int | None = None,
         return_stats: bool = False,
         fill_mode: str = "random",
         middle_mode: str = "origin") -> np.ndarray:
    """
    fill_mode:
      "random" (default) -> PVCS_pairwise.
      "geometric" -> PVCS_perclass : 100% geometric, per-class top-k.

    middle_mode (only applies when fill_mode='random'):
      "origin" (default) -> middle = |proj|≈0 (easy points).
      "midpoint"-> middle = |proj - m|≈0.
    """
    if fill_mode == "geometric":
        return PVCS_perclass(X, labels, target_ratio, edge_ratio,
                             max_neighbor_classes, random_state, return_stats)
    elif fill_mode in ("random", "pairwise", "legacy"):
        return PVCS_pairwise(X, labels, target_ratio, edge_ratio,
                             max_neighbor_classes, random_state, return_stats,
                             middle_mode=middle_mode)
    else:
        raise ValueError(f"PVCS: fill_mode invalid'{fill_mode}' "
                         f"(select 'geometric' or 'random').")


# ==============================================================================
# LOGGING HELPER
# ==============================================================================

_CSV_FIELDS = [
    "dataset", "fraction", "seed", "round", "class",
    "n_samples", "n_selected", "edge", "middle", "random", "trimmed",
    "geometric_share",
]


def _append_selection_stats_csv(csv_path: str, rows: list):
    """Append the stats rows to the CSV (writes the header if the file does not yet exist)."""
    if not csv_path or not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ==============================================================================
# DRIVER: select a coreset with PVCS   (keeps interface + logging; adds pvcs_fill_mode)
# ==============================================================================

def select_pvcs_coreset(
        model,
        full_dataset,
        indices_by_class,
        coreset_size_fraction,
        device,
        gradient_type,
        num_classes,
        edge_ratio,
        max_neighbor_classes,
        random_state,
        gradient_batch_size,
        log_selected_indices=False,
        dataset_name=None,
        seed=None,
        reselection_round=None,
        csv_log_path=None,
        return_selection_stats=False,
        pvcs_fill_mode="random",
        pvcs_middle_mode="origin",
    ):
    """
    Driver for PVCS coreset selection: computes per-class gradients, runs PVCS on the merged gradient matrix, splits the result back per class, computes sample weights.

    :param model: Model used to compute the gradient for each sample.
    :param full_dataset: The full original dataset.
    :param indices_by_class: Dict mapping a class label to its list of indices in full_dataset.
    :param coreset_size_fraction: Coreset size as a fraction of full_dataset.
    :param device: Device (CPU/GPU) used when computing gradients.
    :param gradient_type: Type of gradient representation to compute.
    :param num_classes: Total number of classes in the dataset.
    :param edge_ratio: Fraction r of each class budget spent on edge/extreme samples. Range [0, 1].
    :param max_neighbor_classes: M, the number of nearest neighbor classes (by centroid distance) paired with each class.
    :param random_state: Seed for the random fill.
    :param gradient_batch_size: Batch size when computing gradients.
    :param log_selected_indices: If True, log the selected local/global indices.
    :param dataset_name: Optional dataset name, recorded in the stats log.
    :param seed: Optional seed value recorded in the stats log; falls back to random_state when None.
    :param reselection_round: Optional index of the reselection round, recorded in the stats log.
    :param csv_log_path: Optional path to a CSV file.
    :param return_selection_stats: If True, additionally return the list of per-class stats rows.
    :param pvcs_fill_mode: Selects the algorithm. Default "random".
    :param pvcs_middle_mode: Criterion for the "middle"/core points, applied only when pvcs_fill_mode is "random". "origin" (default).
    
    :return: (final_indices, final_weights, elapsed_time) — global indices of the coreset, their corresponding weights, and the elapsed time (seconds); or
        (final_indices, final_weights, elapsed_time, stats_rows) when return_selection_stats=True.
    """
    print(f">>> Selecting coreset by PVCS... (fill_mode={pvcs_fill_mode})")
    start_time = time.time()
    final_indices, final_weights = [], {}

    classes = [c for c in indices_by_class if len(indices_by_class[c]) > 0]
    if len(classes) == 0:
        warnings.warn("select_pvcs_coreset: Empty coreset.")
        if return_selection_stats:
            return final_indices, final_weights, time.time() - start_time, []
        return final_indices, final_weights, time.time() - start_time

    # Step 1: per-class gradients + merge into X_all/labels_all
    class_gradients_dict = {}
    X_blocks, labels_blocks, offsets = [], [], {}
    offset = 0
    for c in classes:
        class_indices = indices_by_class[c]
        class_gradients = compute_gradient_representations(
            model=model, full_dataset=full_dataset, indices=class_indices,
            device=device, gradient_type=gradient_type, batch_size=gradient_batch_size)
        class_gradients_dict[c] = class_gradients

        X_c = class_gradients.detach().cpu().numpy().astype(np.float64)
        n_c = X_c.shape[0]
        X_blocks.append(X_c)
        labels_blocks.append(np.full(n_c, c))
        offsets[c] = offset
        offset += n_c
        print(f">>> PVCS: calculated gradient for class {c}, samples={n_c}")

    X_all = np.vstack(X_blocks)
    labels_all = np.concatenate(labels_blocks)
    print(f"\n>>> PVCS: Number of samples = {X_all.shape[0]}, number of class = {len(classes)}, "
          f"max_neighbor_classes={max_neighbor_classes}, edge_ratio={edge_ratio}, "
          f"fill_mode={pvcs_fill_mode}, middle_mode={pvcs_middle_mode}")

    # Step 2: run PVCS
    selected_in_Xall, ovo_stats_by_class = PVCS(
        X_all, labels_all,
        target_ratio=coreset_size_fraction,
        edge_ratio=edge_ratio,
        max_neighbor_classes=max_neighbor_classes,
        random_state=random_state,
        return_stats=True,
        fill_mode=pvcs_fill_mode,
        middle_mode=pvcs_middle_mode,
    )
    target_count_total = max(1, int(np.ceil(coreset_size_fraction * X_all.shape[0])))
    print(f">>> PVCS: selected {len(selected_in_Xall)} sample / budget {target_count_total}")

    # Step 3: split by class + compute weights
    selected_local_by_class = {c: [] for c in classes}
    for idx in selected_in_Xall.tolist():
        c = int(labels_all[idx])
        local_idx = idx - offsets[c]
        selected_local_by_class[c].append(local_idx)

    for c in classes:
        selected_local = selected_local_by_class[c]
        class_indices = indices_by_class[c]
        if len(selected_local) == 0:
            print(f"\n>>> PVCS for class {c}: selected 0 points / samples={len(class_indices)}")
            continue
        print(f"\n>>> PVCS for class {c}: selected {len(selected_local)} points "
              f"/ samples={len(class_indices)}")
        if log_selected_indices:
            selected_global = [class_indices[i] for i in selected_local]
            print(f"[PVCS selected indices] class={c}")
            print(f"local_indices : {selected_local}")
            print(f"global_indices: {selected_global}")

        weights_local = calculate_weights(
            class_gradients=class_gradients_dict[c],
            selected_local=selected_local,
        )
        for local_idx, weight in weights_local.items():
            global_idx = class_indices[local_idx]
            final_indices.append(global_idx)
            final_weights[global_idx] = float(weight)

    end_time = time.time()

    # Step 4: console table + collect stats rows
    _seed = seed if seed is not None else random_state
    stats_rows = []

    print("\n>>> PVCS - Summary number of selected samples by class:")
    print(f"{'class':>8}{'samples':>10}{'selected':>10}{'edge':>8}{'middle':>8}{'random':>8}"
          f"{'geom(%)':>10}{'ty_le_lop(%)':>14}{'ty_le_coreset(%)':>18}")
    total_selected = len(final_indices)
    for c in classes:
        n_selected_c = len(selected_local_by_class[c])
        n_samples_c = len(indices_by_class[c])
        s = ovo_stats_by_class.get(c, {})
        n_edge = s.get('edge', 0)
        n_middle = s.get('middle', 0)
        n_random = s.get('random_filled', 0)
        n_trimmed = s.get('trimmed', 0)

        geom_share = (n_edge + n_middle) / n_selected_c if n_selected_c > 0 else 0.0
        ty_le_lop = 100.0 * n_selected_c / n_samples_c if n_samples_c > 0 else 0.0
        ty_le_coreset = 100.0 * n_selected_c / total_selected if total_selected > 0 else 0.0

        print(f"{c!s:>8}{n_samples_c:>10}{n_selected_c:>10}{n_edge:>8}{n_middle:>8}{n_random:>8}"
              f"{100.0*geom_share:>9.2f}%{ty_le_lop:>13.2f}%{ty_le_coreset:>17.2f}%")
        if n_random > 0:
            print(f"         -> class {c}: lack of {n_random} samples, random fill")
        if n_trimmed > 0:
            print(f"         -> class {c}: redandunt {n_trimmed} samples, trimmed")

        stats_rows.append({
            "dataset": dataset_name,
            "fraction": coreset_size_fraction,
            "seed": _seed,
            "round": reselection_round,
            "class": int(c) if isinstance(c, (int, np.integer)) else c,
            "n_samples": n_samples_c,
            "n_selected": n_selected_c,
            "edge": n_edge,
            "middle": n_middle,
            "random": n_random,
            "trimmed": n_trimmed,
            "geometric_share": round(geom_share, 6),
        })

    tot_edge = sum(r["edge"] for r in stats_rows)
    tot_mid = sum(r["middle"] for r in stats_rows)
    overall_geom = (tot_edge + tot_mid) / total_selected if total_selected > 0 else 0.0
    print(f"\n>>> PVCS geometric_share= {100.0*overall_geom:.2f}% "
          f"(edge={tot_edge}, middle={tot_mid}, random={total_selected - tot_edge - tot_mid})")
    print(f">>> PVCS selection done in {end_time - start_time:.2f}s")
    print(f">>> Coreset size: {len(final_indices)}")

    if csv_log_path:
        _append_selection_stats_csv(csv_log_path, stats_rows)
        print(f">>> PVCS: {len(stats_rows)} stats in {csv_log_path}")

    if log_selected_indices:
        print("\n[PVCS final coreset global indices]")
        print(final_indices)

    if return_selection_stats:
        return final_indices, final_weights, end_time - start_time, stats_rows
    return final_indices, final_weights, end_time - start_time