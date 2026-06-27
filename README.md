# HOPE-JEPA-SIGReg

A self-supervised image model that combines three ideas:

1. **HOPE** (Nested Learning) -- Google's backbone where each layer is a
   *Self-Modifying Titans* memory mixer (a neural long-term memory updated by a
   *learned internal optimizer*, full backprop-through-time) plus a *Continuum
   Memory System* of FFN modules at staggered update frequencies.
   *Behrouz et al., "Nested Learning: The Illusion of Deep Learning
   Architectures", arXiv:2512.24695.*
2. **JEPA on every layer** -- at each HOPE layer a predictor head predicts the
   masked-patch embeddings from the context-patch embeddings *in latent space*
   (no pixel reconstruction). Summed across layers this is a deep/hierarchical
   JEPA objective.
   *LeCun, "A Path Towards Autonomous Machine Intelligence"; I-JEPA (Assran et al.).*
3. **SIGReg** -- the Sketched Isotropic-Gaussian regularizer from LeJEPA that
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
will be ~random here -- that's expected; the `tiny` config is for correctness,
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
*without* it. The with-SIGReg probe should beat the without-SIGReg probe -- that
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

## HOPE-JEPA for LLMs (Phase 1: slot-JEPA + QLoRA)

The same HOPE backbone and JEPA/SIGReg objectives are re-purposed for large
language models. `hope_jepa.llm` splices HOPE layers into any HuggingFace
Llama-family model (Qwen2.5, Llama-3, Mistral, Gemma, ...) and adds a **slot
JEPA** auxiliary objective on top of next-token cross-entropy:

* **Slot JEPA** -- K diverging latent slots predict masked BPE positions from
  context, regularized with SIGReg and a slot-diversity term.
* **Optional JEPA-Reasoner** -- a 2-layer latent-rollout + talker module that
  reasons ahead in a compressed space before decoding.
* All new parameters (HOPE layers, slot-JEPA, Reasoner) stay fully trainable; the
  base model can be quantized (4-bit / 8-bit) via `bitsandbytes` + `peft`
  (QLoRA).

Entrypoint:
```bash
# 4-bit QLoRA on a 7B model (GPU required)
python scripts/train_llm_jepa.py --config config/llm_default.yaml \
    --dataset Crownelius/Complete-FABLE.5-traces-2M --output runs/fable_hope
```
Smoke test (CPU, no download):
```bash
python -m tests.test_llm_smoke
```

#### Low-memory optimizers & throughput knobs

`train_llm_jepa.py` ships three memory-efficient optimizers in addition to the
standard AdamW family. These cut optimizer-state and gradient memory so you can
fit a larger batch / sequence (the direct lever on steps/sec):

```bash
# LOMO -- fused SGD, GLOBAL grad clip. Two backward passes per step (a dry pass
# to measure the global norm, then a live pass that applies the clipped update
# and frees each grad as it is produced). No optimizer state at all.
python scripts/train_llm_jepa.py ... --optimizer lomo --lr 1e-5

# AdaLomo (RECOMMENDED) -- momentum + PER-TENSOR local clip, SINGLE backward pass.
# The fastest fused variant; a good fit here because the Titans recurrence makes
# a second backward expensive. Keeps one fp32 momentum buffer per param.
python scripts/train_llm_jepa.py ... --optimizer adalomo --lr 2e-4

# LISA -- Layerwise Importance Sampled AdamW. Activates only --lisa_k of the base
# decoder layers per step (the rest are frozen: no state, no backward into them),
# so optimizer state and backward compute drop by ~(num_layers / k).
python scripts/train_llm_jepa.py ... --optimizer lisa --lisa_k 2 \
    --lisa_refresh_every 50 --lr 2e-4
```

Throughput knobs (`--help` lists all):

* `--prefetch N` (default 4) -- pre-tokenize batches on a CPU thread so the GPU
  never waits on the tokenizer. `0` disables it.
* `--diag_every N` (default 50) -- run the eff-rank SVD + sparsity diagnostic
  only every N steps (it is log-only and forces a GPU sync otherwise).
* `--max_len`, `--target_layers`, `--lisa_k` -- the biggest single-GPU levers on
  steps/sec; lowering any of them proportionally reduces activation memory and
  the cost of the HOPE/Titans recurrence.

> **Note:** LOMO/AdaLomo fuse the update into backward, so they cannot be
> combined with `--grad_accum > 1` (each backward already updates the params).
> The JEPA compute path (mask sampling, slot-JEPA predictor, SIGReg gather) is
> vectorized; `tests/test_optim_and_vectorize.py` is a torch-only gate proving
> the vectorized losses/gradients match the old per-example code to ~1e-7.

---

## EGGROLL self-play RL (Phase 2: adversarial repo repair)

After slot-JEPA pretraining, the center model theta is evolved via **EGGROLL**
adversarial self-play in a real code repository:

* **3 Injectors** introduce subtle bugs into clean repo snapshots.
* **12 Fixers** (6 bare + 6 tool-augmented) race to repair them.
* Validation is against the repo's **real test suite** (SWE-RL style) -- a fix
  only counts if tests go green.
* The center is updated by **rank-r evolution strategies** (EGGROLL,
  arXiv:2511.16652): low-rank perturbations, rank-shaped utilities so the best
  member dominates, no gradients and no critic.
* Reward shaping:
  * **GTPO** (token-level) for injectors -- reward bugs that stump fixers.
  * **GRPO-S** (sequence-level) for fixers -- reward fast, lean repairs with
    few tools and few retries.

Entrypoint:
```bash
python scripts/train_eggroll.py --config config/eggroll_default.yaml \
    --repos ./my_repo --output runs/eggroll/run1
```
CPU smoke (deterministic stub agents, no LLM download):
```bash
python -m tests.test_eggroll_smoke
```

---

## Configs

| config                  | dataset    | use case                                  |
|-------------------------|------------|-------------------------------------------|
| `config/tiny.yaml`      | CIFAR-10   | local correctness / smoke (2k imgs, 3 ep) |
| `config/cifar10_small.yaml` | CIFAR-10 | full data, modest stack, short pretrain |
| `config/cifar100_full.yaml` | CIFAR-100 | full-fidelity, deep stack, long pretrain |
| `config/llm_default.yaml`   | (any HF text) | slot-JEPA + QLoRA on 7B LLM |
| `config/eggroll_default.yaml` | (repo env) | EGGROLL self-play RL phase |

All hyperparameters (HOPE depth, Titans memory width, CMS module count /
cadence, JEPA mask ratio & predictor depth, SIGReg sketch dim / weight) live in
the YAML. The defaults follow the JEPA / Titans literature (mask ratio ~0.6,
predictor depth 2-4, `λ_sig ≈ 1`).

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
hope_jepa/llm/      # Llama-family LLM integration
  wrapper.py      # HopeLLM: HF model + slot-JEPA + Reasoner, unified forward
  surgery.py      # install_hope_layers / build_hope_llm (replace/insert/swap)
  jepa_llm.py     # SlotJEPAForLLM: slot-JEPA + SIGReg + slot-diversity
  reasoner.py     # JepaReasoner: latent rollout + talker
  hope_block.py   # HopeDecoderLayer / MACAttnAdapter shaped as HF decoders
  config.py       # HopeLlmConfig dataclass + layer-spec parser
  step.py         # global step / cadence threading
hope_jepa/rl/        # EGGROLL self-play RL repo-repair system
  eggroll.py      # rank-r ES: sample_perturbation, rank_utilities, eggroll_update
  population.py   # EggrollTrainer: generational self-play loop
  env.py          # RepoEnv: snapshot, apply edits, sandboxed test validation
  roles.py        # Injector / Fixer agents + action parsers
  reward.py       # GTPO injector fitness + GRPO-S fixer fitness shaping
  tools.py        # ToolRegistry: agent-authored tool compile / cache / invoke
  config.py       # EggrollConfig dataclass
scripts/
  train_ssl.py        # SSL pretraining loop for images
  train_llm_jepa.py   # slot-JEPA + QLoRA finetuning for LLMs
  train_eggroll.py    # EGGROLL self-play RL repo-repair evolution
tests/
  test_smoke.py
  test_overfit_batch.py
  test_sigreg.py
  test_shapes.py
  test_llm_smoke.py       # CPU smoke for LLM phase
  test_eggroll_smoke.py   # CPU smoke for EGGROLL RL phase
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
  (caught and fixed during development -- see `test_overfit_batch`).
- SIGReg is computed on a random sketch (`m << d`) for linear time/memory; set
  `sketch_dim: null` for the full-covariance variant.
- **EGGROLL safety**: agents emit code (patches + tools) that is executed inside
  a sandboxed cwd (no network env, CPU/time capped). This is defense-in-depth,
  not a hard boundary -- only point `--repos` at trusted code, and ideally run
  inside a container.

## References
- Behrouz et al., *Nested Learning: The Illusion of Deep Learning Architectures*, arXiv:2512.24695 -- HOPE.
- Behrouz & Hashemi, *Titans: Learning to Memorize at Test Time* -- Self-Modifying memory.
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA).
- Balestriero & LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, arXiv:2511.08544 -- SIGReg.
- Ma et al., *EGGROLL: Rank-r Black-Box Evolution Strategies for Large Language Models*, arXiv:2511.16652 -- EGGROLL optimizer.
- Lv et al., *Full Parameter Fine-tuning for Large Language Models* (LOMO: LOw-Memory Optimization), OpenLMLab/LOMO -- fused update-into-backward.
- Pan et al., *LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning*, arXiv:2403.17919 (NeurIPS 2024) -- LISA optimizer.
- Garg et al., *What Makes or Breaks Policy Optimization?*, arXiv:2508.04349 -- GTPO / GRPO-S reward shaping.
- facebookresearch/swe-rl -- SWE-RL edit-similarity + real test-suite reward.
