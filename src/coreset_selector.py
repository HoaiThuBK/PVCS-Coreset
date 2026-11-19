"""
---file coreset_selector.py---

Dispatcher for coreset selection across multiple methods. Provides a `CoresetSelector` class that routes to the different selection 
strategies (CRAIG, CHVCoreset, Random, CHV-CRAIG) behind a single `select` interface, plus a utility to evaluate coreset quality via the normalized
gradient distance between the weighted coreset gradient and the full-dataset gradient. Also provides `set_seed` for reproducibility and a
`WeightedSubsetDataset` that exposes the selected coreset (with per-sample weights) as a torch Dataset.
"""

import torch
import numpy as np
import random
from torch.utils.data import Dataset
from utils import compute_gradient_representations
from random_selector import select_random_coreset
from src.craig import select_craig_coreset
from gradmatch import select_gradmatch_coreset
from pvcs_coreset import select_pvcs_coreset
from geometry_baselines import select_herding_coreset, select_kcenter_coreset
from el2n import select_el2n_coreset
def set_seed(seed: int):
    """
    Set the seed for the random number generators (random, numpy, torch, cudnn) to make results reproducible.

    :param seed: Seed value used to initialize the random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class CoresetSelector:
    """
    Dispatcher class for coreset selection across several different methods (craig, chvcoreset, random, chv_craig).
    """
    def __init__(
        self,
        model,
        full_dataset,
        indices_by_class,
        coreset_size_fraction,
        device,
        gradient_type,
        num_classes,
        random_state,
        gradient_batch_size,
        pvcs_edge_ratio=0.5,
        pvcs_max_neighbor_classes=4,
        craig_force_cpu=False,
        gradmatch_force_cpu=False,
        kcenter_force_cpu=False,
        el2n_force_cpu = False
    ):
        """
        Initialize and store the parameters shared by all coreset selection methods.

        :param model: Model used to compute the per-sample gradients.
        :param full_dataset: The complete original dataset.
        :param indices_by_class: Dict mapping a class label to a list of indices into full_dataset.
        :param coreset_size_fraction: Coreset size as a fraction of full_dataset.
        :param device: Device (CPU/GPU) used when computing gradients.
        :param gradient_type: Type of gradient representation to compute.
        :param num_classes: Total number of classes in the dataset.
        :param random_state: Seed for the randomized steps in the coreset selection process.
        :param gradient_batch_size: Batch size used when computing gradients.
        :param pvcs_edge_ratio: Edge ratio (r∈[0,1]) of the budget allocated to boundary points for each pair of classes used in the PVCS method
        :param pvcs_max_neighbor_classes: The number of nearest neighboring classes (based on centroids)
        :param craig_force_cpu: If True, the CRAIG greedy loop is always executed on the CPU
        :param kcenter_force_cpu: If True, the k-center is always executed on the CPU
        :param el2n_force_cpu: If True, the EL2N greedy loop is always executed on the CPU
        """
        self.model = model
        self.full_dataset = full_dataset
        self.indices_by_class = indices_by_class
        self.coreset_size_fraction = coreset_size_fraction
        self.device = device
        self.gradient_type = gradient_type
        self.num_classes = num_classes
        self.random_state = random_state
        self.gradient_batch_size = gradient_batch_size
        self.pvcs_edge_ratio = pvcs_edge_ratio
        self.pvcs_max_neighbor_classes = pvcs_max_neighbor_classes
        self.craig_force_cpu = craig_force_cpu
        self.gradmatch_force_cpu = gradmatch_force_cpu
        self.kcenter_force_cpu = kcenter_force_cpu
        self.el2n_force_cpu = el2n_force_cpu

    def select(self, method="craig", **pvcs_log_kwargs):
        """
        Select a coreset using the specified method.

        :param method: Name of the coreset selection method (craig, random, pvcs, herding, kcenter, el2n, gradmatch).
        :return: The value returned by the select function corresponding to the chosen method.
        """
        if method == "craig":
            return select_craig_coreset(
                self.model,
                self.full_dataset,
                self.indices_by_class,
                self.coreset_size_fraction,
                self.device,
                self.gradient_type,
                self.num_classes,
                self.gradient_batch_size,
                force_cpu=self.craig_force_cpu,
            )
        elif method == "gradmatch":
            return select_gradmatch_coreset(
                self.model,
                self.full_dataset,
                self.indices_by_class,
                self.coreset_size_fraction,
                self.device,
                self.gradient_type,
                self.num_classes,
                self.gradient_batch_size,
                force_cpu=self.gradmatch_force_cpu,
            )
        elif method == "random":
            return select_random_coreset(
                self.model,
                self.full_dataset,
                self.indices_by_class,
                self.coreset_size_fraction,
                self.device,
                self.gradient_type,
                self.num_classes,
                self.random_state,
                self.gradient_batch_size,
            )
        elif method == "pvcs":
            return select_pvcs_coreset(
                self.model,
                self.full_dataset,
                self.indices_by_class,
                self.coreset_size_fraction,
                self.device,
                self.gradient_type,
                self.num_classes,
                self.pvcs_edge_ratio,
                self.pvcs_max_neighbor_classes,
                self.random_state,
                self.gradient_batch_size,
                **pvcs_log_kwargs, 
            )
        elif method == "herding":
            return select_herding_coreset(
                self.model,
                self.full_dataset,
                self.indices_by_class,
                self.coreset_size_fraction,
                self.device,
                self.gradient_type,
                self.num_classes,
                self.gradient_batch_size,
            )
        elif method == "kcenter":
            return select_kcenter_coreset(
                self.model,
                self.full_dataset,
                self.indices_by_class,
                self.coreset_size_fraction,
                self.device,
                self.gradient_type,
                self.num_classes,
                self.gradient_batch_size,
                random_state=self.random_state,
                force_cpu=self.kcenter_force_cpu,
            )
        elif method == "el2n":
            return select_el2n_coreset(
                self.model,
                self.full_dataset,
                self.indices_by_class,
                self.coreset_size_fraction,
                self.device,
                self.gradient_type,
                self.num_classes,
                self.gradient_batch_size,
                force_cpu=self.el2n_force_cpu,
            )
        else:
            raise ValueError(f"Phương pháp lựa chọn không hợp lệ: {method}")
        
    def compute_grad_dist_for_current_coreset(self, coreset_indices, coreset_weights):
        """
        Compute the normalized gradient distance between the weighted total gradient of the coreset and the total gradient of the full dataset, used to assess coreset quality.

        :param coreset_indices: List of global indices of the samples in the coreset.
        :param coreset_weights: Dict mapping a global index to the weight of the corresponding sample in the coreset.
        :return: The normalized gradient distance (float); returns 0.0 if the coreset is empty.
        """
        if not coreset_indices or not coreset_weights:
            return 0.0
        total_full_gradient = None
        total_coreset_gradient = None
        for c in range(self.num_classes):
            class_indices = self.indices_by_class.get(c, [])
            if len(class_indices) == 0:
                continue
            class_gradients = compute_gradient_representations(
                model=self.model,
                full_dataset=self.full_dataset,
                indices=class_indices,
                device=self.device,
                gradient_type=self.gradient_type,
                batch_size=self.gradient_batch_size,
            )
            if class_gradients.shape[0] == 0:
                continue
            if total_full_gradient is None:
                grad_dim = class_gradients.shape[1]
                total_full_gradient = torch.zeros(grad_dim, dtype=torch.float32)
                total_coreset_gradient = torch.zeros(grad_dim, dtype=torch.float32)
            total_full_gradient += class_gradients.sum(dim=0)
            global_to_local = {
                g_idx: i for i, g_idx in enumerate(class_indices)
            }
            class_coreset_local = []
            class_weights = []
            for g_idx in coreset_indices:
                if g_idx in global_to_local:
                    class_coreset_local.append(global_to_local[g_idx])
                    class_weights.append(float(coreset_weights.get(g_idx, 1.0)))
            if len(class_coreset_local) == 0:
                continue
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            coreset_class_gradient = torch.sum(
                class_gradients[class_coreset_local] * weights_tensor.unsqueeze(1),
                dim=0,
            )
            total_coreset_gradient += coreset_class_gradient
        if total_full_gradient is None:
            return 0.0
        gradient_distance = torch.norm(total_full_gradient - total_coreset_gradient).item()
        full_gradient_norm = torch.norm(total_full_gradient).item()
        normalized_distance = gradient_distance / (full_gradient_norm + 1e-8)
        return normalized_distance

class WeightedSubsetDataset(Dataset):
    """
    Dataset wrapper around full_dataset that contains only the samples belonging to the coreset, together with their corresponding weights.
    """
    def __init__(self, full_dataset, indices, weights):
        """
        Initialize and store the original dataset along with the coreset's list of indices and weights.

        :param full_dataset: The complete original dataset.
        :param indices: List of global indices of the samples belonging to the coreset.
        :param weights: Dict mapping a global index to the weight of the corresponding sample in the coreset.
        """
        self.full_dataset = full_dataset
        self.indices = indices
        self.weights = weights
    def __len__(self):
        """
        Return the number of samples in the coreset.

        :return: The number of elements in self.indices.
        """
        return len(self.indices)
    def __getitem__(self, idx):
        """
        Get one sample (data, label, weight) by its local index within the coreset.

        :param idx: Local index (position in self.indices) of the sample to retrieve.
        :return: A tuple (x, y, weight) of the input data, the label, and the weight (as a tensor).
        """
        global_idx = self.indices[idx]
        x, y = self.full_dataset[global_idx]
        weight = self.weights.get(global_idx, 1.0)
        return x, y, torch.tensor(weight, dtype=torch.float32)
