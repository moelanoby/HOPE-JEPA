"""Data pipeline for HOPE-JEPA-SIGReg on CIFAR.

Two responsibilities:
  1. Build (possibly subsetted) CIFAR-10/100 train/val dataloaders. SSL training
     ignores labels (returns two augmented views of each image); the linear
     probe uses labels.
  2. Patchify an image batch into a token sequence, and generate random JEPA
     context/target masks shared across all HOPE layers.

We deliberately keep augmentation lightweight (random crop + flip + normalize),
matching the standard CIFAR SSL recipe. All randomness is seedable.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------------------
# Transforms / datasets
# ---------------------------------------------------------------------------
def ssl_transform():
    """Augmentation applied independently to each of the two JEPA views."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def eval_transform():
    """Deterministic transform for the linear-probe / evaluation pipeline."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def _make_dataset(dataset: str, root: str, train: bool, transform):
    if dataset == "cifar10":
        return datasets.CIFAR10(root=root, train=train, transform=transform, download=True)
    if dataset == "cifar100":
        return datasets.CIFAR100(root=root, train=train, transform=transform, download=True)
    raise ValueError(f"Unknown dataset {dataset!r}; expected 'cifar10' or 'cifar100'.")


def _maybe_subset(ds, n):
    """Deterministically subsample to `n` examples (seeded)."""
    if n is None:
        return ds
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    return Subset(ds, idx.tolist())


def num_classes(dataset: str) -> int:
    return 100 if dataset == "cifar100" else 10


# ---------------------------------------------------------------------------
# A tiny Dataset wrapper that yields two augmented views (for SSL).
# ---------------------------------------------------------------------------
class TwoViewDataset(torch.utils.data.Dataset):
    """Wraps an image dataset and returns two independently augmented views."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        # torchvision transforms with RandomCrop/Flip are stochastic per call,
        # so calling __getitem__ twice yields two different views.
        view1, _ = self.base[i]
        view2, _ = self.base[i]
        return view1, view2, label


# ---------------------------------------------------------------------------
# Dataloader builders
# ---------------------------------------------------------------------------
def build_ssl_loaders(cfg):
    """Returns (train_loader, val_loader) for SSL pretraining (two views, no
    labels consumed by the loss but kept for diagnostics)."""
    d = cfg["data"]
    tfm = ssl_transform()
    train_ds = _make_dataset(d["dataset"], d["root"], train=True, transform=tfm)
    val_ds = _make_dataset(d["dataset"], d["root"], train=False, transform=tfm)
    train_ds = _maybe_subset(train_ds, d.get("subset_train"))
    val_ds = _maybe_subset(val_ds, d.get("subset_val"))

    g = torch.Generator().manual_seed(cfg.get("seed", 0))
    train_loader = DataLoader(
        TwoViewDataset(train_ds),
        batch_size=cfg["ssl"]["batch_size"], shuffle=True,
        num_workers=d["num_workers"], drop_last=True, generator=g,
    )
    val_loader = DataLoader(
        TwoViewDataset(val_ds),
        batch_size=cfg["ssl"]["batch_size"], shuffle=False,
        num_workers=d["num_workers"],
    )
    return train_loader, val_loader


def build_probe_loaders(cfg):
    """Returns (train_loader, val_loader) for the frozen-backbone linear probe.
    Uses deterministic transforms and returns (image, label)."""
    d = cfg["data"]
    tfm = eval_transform()
    train_ds = _make_dataset(d["dataset"], d["root"], train=True, transform=tfm)
    val_ds = _make_dataset(d["dataset"], d["root"], train=False, transform=tfm)
    train_ds = _maybe_subset(train_ds, d.get("subset_train"))
    val_ds = _maybe_subset(val_ds, d.get("subset_val"))

    g = torch.Generator().manual_seed(cfg.get("seed", 0))
    train_loader = DataLoader(
        train_ds, batch_size=cfg["probe"]["batch_size"], shuffle=True,
        num_workers=d["num_workers"], drop_last=True, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["probe"]["batch_size"], shuffle=False,
        num_workers=d["num_workers"],
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Patchify + masking
# ---------------------------------------------------------------------------
def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[B, C, H, W] -> [B, N, C*P*P] flattened patch embeddings (N = (H/P)*(W/P)).

    Patches are ordered row-major over the patch grid. Each image is split into
    non-overlapping patch_size x patch_size blocks and flattened per patch.
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, "image must be divisible by patch size"
    ph = pw = patch_size
    nh, nw = H // ph, W // pw
    # [B, C, nh, ph, nw, pw] -> [B, nh, nw, ph, pw, C] -> [B, nh*nw, ph*pw*C]
    x = images.reshape(B, C, nh, ph, nw, pw)
    x = x.permute(0, 2, 4, 3, 5, 1)             # [B, nh, nw, ph, pw, C]
    x = x.reshape(B, nh * nw, ph * pw * C)
    return x


def random_mask(batch_size: int, num_tokens: int, mask_ratio: float,
                device, generator=None) -> torch.Tensor:
    """Returns a boolean mask [B, N]; True == target/masked token.

    For each example, independently sample ~mask_ratio fraction of tokens to be
    the JEPA targets (predicted from the remaining context tokens).
    """
    n_mask = int(round(num_tokens * mask_ratio))
    n_mask = max(1, min(n_mask, num_tokens - 1))  # keep at least 1 of each
    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    for i in range(batch_size):
        idx = torch.randperm(num_tokens, generator=generator, device=device)[:n_mask]
        mask[i, idx] = True
    return mask
