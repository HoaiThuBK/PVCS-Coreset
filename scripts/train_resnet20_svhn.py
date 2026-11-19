"""
--- file train_resnet20_svhn.py ---

Training script for a ResNet-20 model on SVHN (10 classes), with support for several coreset selection 
methods (craig, random, pvcs, herding, kcenter, el2n, gradmatch) or training on the full dataset.
"""
import numpy as np
import torch
import csv
import os
import sys
import argparse
import random
import torch.nn as nn
import torch.optim as optim


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import load_svhn_all, get_indices_by_class, train_one_epoch, evaluate
from model_resnet import resnet20
from src.coreset_selector import CoresetSelector, WeightedSubsetDataset

def set_seed(seed):
    """
    Seed all RNGs and enable deterministic execution for reproducibility.

    Enables PyTorch deterministic algorithms (use_deterministic_algorithms) and disables TF32 to reduce floating-point discrepancies across different
    GPU types. This does not guarantee bit-for-bit reproducibility across GPU architectures, but removes the noise that is within the code's control.

    :param seed: Seed value used to initialize the random number generators.
    """
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _capture_rng_state(g):
    """
    Capture the current state of all random sources for the warm-up checkpoint.

    :param g: torch.Generator used by full_trainloader (warm-up shuffle).
    :return: dict of RNG states, to be stored as part of the checkpoint.
    """
    return {
        'rng_python': random.getstate(),
        'rng_numpy':  np.random.get_state(),
        'rng_torch':  torch.get_rng_state(),
        'rng_cuda':   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'rng_g':      g.get_state(),
    }


def _restore_rng_state(checkpoint, g):
    """
    Restore RNG state from a checkpoint. Guarded on the presence of the 'rng_torch' key for backward compatibility with old checkpoints.

    :param checkpoint: checkpoint dict loaded via torch.load.
    :param g: torch.Generator of full_trainloader, whose state is reset here.
    :return: True if RNG state was restored; False if an old checkpoint contains no RNG state.
    """
    if 'rng_torch' not in checkpoint:
        return False
    random.setstate(checkpoint['rng_python'])
    np.random.set_state(checkpoint['rng_numpy'])
    torch.set_rng_state(checkpoint['rng_torch'].cpu() if hasattr(checkpoint['rng_torch'], 'cpu')
                        else checkpoint['rng_torch'])
    rng_cuda = checkpoint.get('rng_cuda')
    if rng_cuda is not None and torch.cuda.is_available():
        try:
            rng_cuda = [s.cpu() if hasattr(s, 'cpu') else s for s in rng_cuda]
            torch.cuda.set_rng_state_all(rng_cuda)
        except Exception as e:
            print(f"CWarning: Unable recover RNG CUDA({e}) ")
    if 'rng_g' in checkpoint:
        rng_g = checkpoint['rng_g']
        g.set_state(rng_g.cpu() if hasattr(rng_g, 'cpu') else rng_g)
    return True


def main():
    """
    Main function: parse command-line arguments, prepare the data/model/optimizer, then run the training loop over epochs (warm-up on the full dataset, then
    periodically re-select the coreset and train with weights), while logging the results (loss, accuracy, time, ...) to a CSV file.
    """
    parser = argparse.ArgumentParser(description= 'Coreset training with Resnet20 on SVHN 10 lớp')

    # --- Define the command-line arguments ---
    parser.add_argument('--selection_method', type=str, default='pvcs',
                        choices=['craig', 'full_dataset', 'random','pvcs','herding','kcenter','moderate','el2n','gradmatch'],
                        help='Coreset selection method, or full-dataset training')
    parser.add_argument('--coreset_fraction', type=float, default=0.01,
                        help='Coreset fraction (e.g. 0.1, 0.3)')
    parser.add_argument('--update_freq', type=int, default=10,
                        help='Coreset update frequency (in epochs)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Total number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Mini-batch size for a full-dataset epoch (warm-up/full_dataset)')
    parser.add_argument('--coreset_batch_size', type=int, default=128,
                        help='Mini-batch size when training on a coreset')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='Number of warm-up epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--gradient_type', type=str, default="logit", choices=["logit", "embedding"],
                        help='Gradient representation method')
    parser.add_argument('--coreset_lr_scale', type=float, default=0.2,
                    help='HLR reduction factor when switching to coreset training')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                    help='Learning rate decay factor at 50% and 75% of the epochs')
    parser.add_argument('--warmup_checkpoint_dir', type=str, default="results_svhn/checkpoints",
                    help='Directory for saving/loading warm-up checkpoints, shared across all selection_methods')
    parser.add_argument('--pvcs_log_csv', type=str, default='results_svhn/pvcs/',
                    help='CSV recording the PVCS coreset component for each re-selection round')
    parser.add_argument('--pvcs_edge_ratio', type=float, default=0.5,
                        help='PVCS: edge ratio r∈[0,1]')
    parser.add_argument('--pvcs_max_neighbor_classes', type=int, default=2,
                        help='PVCS: nearest neighbors M')
    args = parser.parse_args()
    set_seed(args.seed)
    print(f"Sử dụng seed: {args.seed}")


    # --- Set up the environment ---
    if torch.backends.mps.is_available(): device = 'mps'
    elif torch.cuda.is_available(): device = 'cuda'
    else: device = 'cpu'
    print(f"Device: {device}")

    # --- Load the SVHN ---
    trainset_full, testset = load_svhn_all()
    trainset_for_selection = trainset_full
    num_classes = 10
    input_dim = None
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Initialize the model ---
    print(f"Model: Resnet20 ({input_dim} features)")
    model = resnet20(num_classes=10).to(device)

    # --- Get indices by class (used by the per-class coreset selection methods) ---
    indices_by_class = get_indices_by_class(trainset_for_selection, num_classes=10)

    # --- Optimizer (Only SGD is used for this experiment) ---
    l2_reg = 1e-4
    print(f"Optimizer: SGD (LR={args.lr}, L2 Reg={l2_reg})")
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=l2_reg)

    def get_current_lr(epoch, args):
        """
        Compute the current learning rate as a function of the epoch: linear warm-up, then reduce the LR if training on a coreset, and reduce it
        further at 50%/75% of the total number of epochs (in place of a torch LR scheduler).

        :param epoch: Current epoch index (starting from 0).
        :param args: Command-line argument namespace (contains lr, warmup_epochs, epochs, ...).
        :return: The learning rate value to apply for the current epoch.
        """
        current_epoch = epoch + 1

        # Warm-up: increase the LR linearly
        if current_epoch <= args.warmup_epochs:
            return args.lr * current_epoch / args.warmup_epochs

        # After warm-up: reduce the LR if training on a coreset
        lr = args.lr

        if args.selection_method != "full_dataset":
            lr = lr * args.coreset_lr_scale

        # Reduce the LR at 50% and 75% of the total number of epochs
        milestone_1 = int(args.epochs * 0.5)
        milestone_2 = int(args.epochs * 0.75)

        if current_epoch >= milestone_2:
            lr = lr * (args.lr_gamma ** 2)
        elif current_epoch >= milestone_1:
            lr = lr * args.lr_gamma

        return lr

    # --- Initialize the coreset selector (shared by every selection_method) ---
    selector = CoresetSelector(
        model=model,
        full_dataset=trainset_for_selection,
        indices_by_class=indices_by_class,
        coreset_size_fraction=args.coreset_fraction,
        device=device,
        gradient_type=args.gradient_type,
        gradient_batch_size=args.batch_size,
        num_classes=num_classes,
        random_state = args.seed,
        pvcs_edge_ratio = args.pvcs_edge_ratio,
        pvcs_max_neighbor_classes = args.pvcs_max_neighbor_classes
    )

    # --- Set up saving to the .csv file ---
    output_dir = os.path.join("results_svhn", args.selection_method)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"results_svhn_resnet20_{args.selection_method}_{args.gradient_type}_seed{args.seed}_r{args.pvcs_edge_ratio}_M{args.pvcs_max_neighbor_classes}"

    if args.selection_method != 'full_dataset':
        filename += f"_frac{args.coreset_fraction}"

    filename += ".csv"

    filepath = os.path.join(output_dir, filename)

    print(f"Sẽ lưu kết quả vào: {filepath}")

    csv_file = open(filepath, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'loss', 'accuracy', 'lr', 'train_time_s', 'selection_time_s','coreset_size'])

    coreset_indices, coreset_weights = None, None
    pvcs_round = 0  
    current_coreset_size = 0
    g = torch.Generator()
    g.manual_seed(args.seed)
    full_trainloader = torch.utils.data.DataLoader(trainset_full, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=g)
    
    # --- Determine the warm-up checkpoint path (automatically based on the seed) and check for its existence --- 
    start_epoch = 0
    warmup_rows_cache = []
    checkpoint_path = None

    if args.warmup_checkpoint_dir and args.selection_method != 'full_dataset':
        os.makedirs(args.warmup_checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.warmup_checkpoint_dir, f"warmup_seed{args.seed}.pt")

        if os.path.exists(checkpoint_path):
            print(f"\n>>> Found checkpoint warm-up: {checkpoint_path} ->  automatic reload.")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if checkpoint.get('seed') != args.seed:
                print(f"Warning: seed in checkpoint ({checkpoint.get('seed')}) different with --seed "
                      f"current ({args.seed}) — The results may not be reproducible")
            if checkpoint.get('warmup_epochs') != args.warmup_epochs:
                print(f"Warning: warmup_epochs in checkpoint ({checkpoint.get('warmup_epochs')}) "
                      f"different with -- current warmup_epochs ({args.warmup_epochs}) — LR schedule may "
                      f"be inconsistent with the original warm-up run")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if _restore_rng_state(checkpoint, g):
                print(">>> RNG state restored from checkpoint (bit-for-bit reconstruction following warm-up)")
            else:
                print(">>>  Warning: The old checkpoint does NOT contain the RNG state")

            warmup_rows_cache = checkpoint['warmup_log_rows']
            start_epoch = checkpoint['epoch']
            print(f">>> Done reload (epoch complete: {start_epoch}). Skip warm-up, "
                  f"continue training from epoch {start_epoch + 1} with selection_method="
                  f"'{args.selection_method}'.\n")
        else:
            print(f"\n>>> No warm-up checkpoint for the seed {args.seed} tại: {checkpoint_path}")
            print(f">>> Start warm-up run and then automatically save the data for use in future runs.\n")

        # Copy original log warm-up to the CSV log
        for row in warmup_rows_cache:
            csv_writer.writerow(row)
        csv_file.flush()

    # --- Main training loop ---
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        current_lr = get_current_lr(epoch, args)

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        print(f">>> Current LR: {current_lr:.6f}")
        selection_time = 0.0

        if args.selection_method == 'full_dataset':
            # Baseline mode: always train on the full dataset, no coreset selection.
            print("Huấn luyện trên toàn bộ dataset...")
            trainloader = full_trainloader
            use_weights_in_loss = False
            selection_time = 0.0
            current_coreset_size = len(trainset_full)

        elif epoch < args.warmup_epochs:
            # Warm-up phase: always use the full dataset so the model stabilizes before selecting a coreset.
            print(f"Warm-up phase (Epoch {epoch+1}/{args.warmup_epochs}): Training on full dataset...")
            trainloader = full_trainloader
            use_weights_in_loss = False
            selection_time = 0.0
            current_coreset_size = len(trainset_full)
        else:
            # After warm-up: weighted training, periodically (every update_freq epochs) re-select the coreset.
            use_weights_in_loss = True
            if (epoch - args.warmup_epochs) % args.update_freq == 0:
                print(f"Warm-up complete. Updating coreset at epoch {epoch+1}...")
                select_kwargs = {}
                if args.selection_method == 'pvcs':
                    select_kwargs = dict(
                        dataset_name='SVHN',
                        seed=args.seed,
                        reselection_round=pvcs_round,
                        csv_log_path=f"{args.pvcs_log_csv}pvcs_composition_r{args.pvcs_edge_ratio}.csv",
                    )
                    pvcs_round += 1
                result = selector.select(method=args.selection_method, **select_kwargs)
                if len(result) == 4:
                    coreset_indices, coreset_weights, selection_time, _ = result
                else:
                    coreset_indices, coreset_weights, selection_time = result
                current_coreset_size = len(coreset_indices) if coreset_indices is not None else 0

                print(f"Number of samples coreset: {current_coreset_size}")

                if not coreset_indices:
                    # Empty coreset: log this epoch with zero values and skip it, no training.
                    print("Warning: Coreset is empty, skipping this epoch.")
                    csv_writer.writerow([epoch+1, 0, 0, optimizer.param_groups[0]['lr'], 0, selection_time, current_coreset_size])
                    continue
            else:
                # Not an update epoch: reuse the coreset selected in a previous epoch.
                selection_time = 0.0

            if not coreset_indices:
                 # No valid coreset has ever been selected: temporarily fall back to the full dataset.
                 print("Warning: coreset_indices is empty, using the full dataset for this epoch.")
                 trainloader = full_trainloader
                 use_weights_in_loss = False
            else:
                coreset_dataset = WeightedSubsetDataset(trainset_full, coreset_indices, coreset_weights)

                g_coreset = torch.Generator()
                g_coreset.manual_seed(args.seed)
                effective_coreset_batch_size = args.coreset_batch_size or len(coreset_dataset)
                trainloader = torch.utils.data.DataLoader(coreset_dataset, batch_size=effective_coreset_batch_size, shuffle=True, num_workers=0, generator=g_coreset)

        # --- Train for one epoch and evaluate on the test set ---
        train_loss, train_time = train_one_epoch(model, trainloader, criterion, optimizer, device, use_weights=use_weights_in_loss)
        accuracy = evaluate(model, testloader, device)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={accuracy:.2f}%, LR={current_lr:.5f}, Time(Train)={train_time:.2f}s, Time(Select)={selection_time:.2f}s")

        row = [epoch+1, f"{train_loss:.4f}", f"{accuracy:.2f}", f"{current_lr:.5f}", f"{train_time:.2f}", f"{selection_time:.2f}", current_coreset_size]
        csv_writer.writerow(row)
        csv_file.flush()

        if args.selection_method != 'full_dataset' and epoch < args.warmup_epochs:
            warmup_rows_cache.append(row)

        # --- Finish warm-up -> auto save checkpoint --- 
        if (checkpoint_path
                and args.selection_method != 'full_dataset'
                and epoch + 1 == args.warmup_epochs):
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'warmup_log_rows': warmup_rows_cache,
                'seed': args.seed,
                'warmup_epochs': args.warmup_epochs,
            }
            checkpoint.update(_capture_rng_state(g))
            torch.save(checkpoint, checkpoint_path)
            print(f">>> Saved checkpoint warm-up (epoch {epoch+1}, include RNG state in: {checkpoint_path}")

    csv_file.close()
    print(f"\nTraining complete. Results have been saved to file: {filename}")


if __name__ == '__main__':
    main()