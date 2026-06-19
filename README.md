# HOPE-JEPA-SIGReg

A self-supervised image model that combines three ideas:

1. **HOPE** (Nested Learning) — Google's backbone where each layer is a
   *Self-Modifying Titans* memory mixer (a neural long-term memory updated by a
   *learned internal optimizer*, full backprop-through-time) plus a *Continuum
   Memory System* of FFN modules at staggered update frequencies.
   *Behrouz et al., "Nested Learning: The Illusion of Deep Learning
   Architectures", arXiv:2512.24695.*
2. **JEPA on every layer** — at each HOPE layer a predictor head predicts the
   masked-patch embeddings from the context-patch embeddings *in latent space*
   (no pixel reconstruction). Summed across layers this is a deep/hierarchical
   JEPA objective.
   *LeCun, "A Path Towards Autonomous Machine Intelligence"; I-JEPA (Assran et al.).*
3. **SIGReg** — the Sketched Isotropic-Gaussian regularizer from LeJEPA that
   pushes the embedding covariance toward `sigma^2 * I`. This is the linchpin:
   per-layer JEPA that targets the network's own embeddings is collapse-prone
   by default, and SIGReg is the documented fix.
   *Balestriero & LeCun, "LeJEPA", arXiv:2511.08544.*

Training is **self-supervised** (no labels): pretrain the backbone with
`Σ_l JEPA^l + λ·SIGReg`, then evaluate by freezing the backbone and training a
single linear classifier (the canonical JEPA linear probe).

---

## Quick start

```bash
pip install -r requirements.txt
```

### 1. Verify correctness locally (CPU/GPU, a few minutes)
```bash
bash scripts/run_tiny.sh
```
Runs the test suite, then a 3-epoch SSL pretrain on 2000 CIFAR-10 images, then
the linear probe. This proves the whole pipeline is wired correctly. Accuracy
will be ~random here — that's expected; the `tiny` config is for correctness,
not performance.

### 2. Run the individual tests
```bash
python -m tests.test_shapes         # shapes + BPTT gradient connectivity
python -m tests.test_overfit_batch  # model can learn + SIGReg prevents collapse
python -m tests.test_sigreg         # SIGReg pushes collapsed rank 2 -> d
python -m tests.test_smoke          # end-to-end finite, dimensionally correct
```

### 3. Run for a real result (cloud / multi-GPU)
```bash
# modest: full CIFAR-10
CONFIG=config/cifar10_small.yaml bash scripts/run_cloud.sh

# heavy: full CIFAR-100, deep stack, long pretrain
CONFIG=config/cifar100_full.yaml bash scripts/run_cloud.sh
```
`run_cloud.sh` runs the **SIGReg ablation**: pretrain + probe *with* SIGReg and
*without* it. The with-SIGReg probe should beat the without-SIGReg probe — that
gap is the headline result (SIGReg prevents collapse).

---

## The key idea, and what the logs show you

Per-layer JEPA predicting the net's own embeddings is, by itself,
representation-collapse-prone (the trivial optimum is a constant embedding).
SIGReg fixes exactly this. The training logs print three numbers each step:

| metric    | meaning                                                |
|-----------|--------------------------------------------------------|
| `jepa`    | mean per-layer prediction loss (should decrease)       |
| `sigreg`  | the covariance penalty (held away from 0 = healthy)    |
| `eff_rank`| effective rank of the embedding covariance (~d healthy, ~1 collapsed) |

The `test_overfit_batch` test demonstrates this concretely on a single batch:
with weak SIGReg the rank collapses to ~1, with adequate SIGReg it stays ~25
*while JEPA still learns*. That is the entire thesis of this codebase in one run.

---

## Configs

| config                  | dataset    | use case                                  |
|-------------------------|------------|-------------------------------------------|
| `config/tiny.yaml`      | CIFAR-10   | local correctness / smoke (2k imgs, 3 ep) |
| `config/cifar10_small.yaml` | CIFAR-10 | full data, modest stack, short pretrain |
| `config/cifar100_full.yaml` | CIFAR-100 | full-fidelity, deep stack, long pretrain |

All hyperparameters (HOPE depth, Titans memory width, CMS module count /
cadence, JEPA mask ratio & predictor depth, SIGReg sketch dim / weight) live in
the YAML. The defaults follow the JEPA / Titans literature (mask ratio ~0.6,
predictor depth 2–4, `λ_sig ≈ 1`).

---

## Layout

```
hope_jepa/
  titans.py   # Self-Modifying memory mixer: learned optimizer + surprise-gated
              # L2 self-update of a [d,d] memory, full BPTT. MAC fusion.
  hope.py     # HOPE layer = Titans mixer + Continuum Memory System
              # (FFN modules at staggered update frequencies 2^k).
  jepa.py     # per-layer predictor + context/target masking + layer loss.
  sigreg.py   # sketched isotropic-gaussian covariance regularizer + eff-rank.
  model.py    # patch-embed + L x HOPE + per-layer JEPA heads + SIGReg.
  losses.py   # aggregate SSL loss (Σ_l JEPA + λ SIGReg) + diagnostics.
  data.py     # CIFAR loaders, augmentations, patchify, two-view, masking.
  utils.py    # config load, seeding, cosine LR, device.
train_ssl.py     # SSL pretraining loop (config-driven, --no-sigreg ablation).
linear_probe.py  # freeze backbone, train linear head, report top-1 accuracy.
tests/           # shapes, overfit-batch, sigreg, smoke.
scripts/         # run_tiny.sh (local), run_cloud.sh (real run + ablation).
config/          # tiny / cifar10_small / cifar100_full YAMLs.
```

---

## Notes & honesty

- **Full-fidelity HOPE** (real learned-optimizer memory recurrence + multi-
  frequency CMS) is compute-heavy. The `tiny` config exists to prove correctness
  on weak hardware, *not* to reach final accuracy. Real numbers require
  `cifar10_small` or `cifar100_full` on the cloud.
- The Titans memory update gate is sigmoid-bounded so `alpha ∈ (0,1)`; this is
  both faithful (Titans' `g` is bounded) and necessary for numerical stability
  across the BPTT recurrence. Without it the memory diverges within ~30 steps
  (caught and fixed during development — see `test_overfit_batch`).
- SIGReg is computed on a random sketch (`m ≪ d`) for linear time/memory; set
  `sketch_dim: null` for the full-covariance variant.

## References
- Behrouz et al., *Nested Learning: The Illusion of Deep Learning Architectures*, arXiv:2512.24695 — HOPE.
- Behrouz & Hashemi, *Titans: Learning to Memorize at Test Time* — Self-Modifying memory.
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA).
- Balestriero & LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, arXiv:2511.08544 — SIGReg.
