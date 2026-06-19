"""Self-Modifying Titans memory mixer (full-fidelity, BPTT).

Implements the core of Google's Nested-Learning memory module: a *neural
long-term memory* matrix `M` that is updated *in-place across the token
sequence* by a *learned internal optimizer*. Unlike a cached/stop-gradient
associative memory, gradients here flow through the entire recurrence
(backprop-through-time, BPTT), so the memory-update rule itself is learned.

Reference structure (Titans MAC variant):
  surprise_t  = theta1 * g1(query) + theta2 * g2(retrieval) + theta3 * g3(stimulus)
  lr_t        = sigmoid(MLP_lr([query, retrieval, stimulus]))   # learned, per-step lr
  alpha_t     = surprise_t * lr_t                                # update gate
  M_t         = (1 - alpha_t) * M_{t-1} + alpha_t * stimulus     # L2-style self-update
  out_t       = MAC: gated fuse of persistent-memory, hidden-state, retrieval

where `stimulus` is an outer-product-style memory write derived from the token.
We expose `d_hidden` for the MLP widths. All tensor ops are batched over the
sequence dimension so the recurrence is a single loop over tokens with BPTT.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _erf_g(x: torch.Tensor) -> torch.Tensor:
    """g(.) nonlinearity from Titans: erf(x/sqrt(2))/2 + 0.5 in [-?]. We use the
    smooth, bounded form `sigmoid` for numerical stability; the paper's exact g
    is erf-based but sigmoid is a standard stable surrogate."""
    return torch.sigmoid(x)


class NeuralLongTermMemory(nn.Module):
    """A neural long-term memory with a learned internal optimizer.

    The memory M is a learnable [d, d] matrix (initialized once). For each token
    we compute a surprise score and a learned learning rate, then perform an
    L2-style self-update that is fully differentiable through time.

    Args:
        d_model: token embedding dim (memory is [d_model, d_model]).
        d_hidden: width of the surprise / lr MLPs.
        init_memory_std: std of the initial memory matrix.
    """

    def __init__(self, d_model: int, d_hidden: int, init_memory_std: float = 0.02):
        super().__init__()
        self.d_model = d_model
        # Learnable initial memory state. We clone this per forward pass so the
        # recurrence can modify it without corrupting the learned init.
        self.M0 = nn.Parameter(torch.randn(d_model, d_model) * init_memory_std)

        # Learned "internal optimizer" weights. theta controls surprise mixing;
        # the lr-MLP produces a per-step, per-token learning rate.
        self.theta = nn.Parameter(torch.tensor([0.33, 0.33, 0.34]))
        # g1(query), g2(retrieval), g3(stimulus): small linear projections.
        self.g1 = nn.Linear(d_model, d_hidden)
        self.g2 = nn.Linear(d_model, d_hidden)
        self.g3 = nn.Linear(d_model, d_hidden)
        # surprise combiner -> scalar per token.
        self.surprise_head = nn.Linear(d_hidden, 1, bias=False)
        # learned lr net: takes [query | retrieval | stimulus] -> scalar in (0,1).
        self.lr_net = nn.Sequential(
            nn.Linear(3 * d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1),
        )
        # Per-token decay (forget) gate, learned.
        self.decay_net = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_model),
        )

    def _stimulus(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Outer-product-style memory write stimulus [B, d, d].

        For a sequence token we write `value key^T` (a rank-1 update), which is
        the associative-memory / delta-rule flavor used in Titans. We scale by
        1/sqrt(d) to keep the update magnitude stable as d grows.
        """
        B, d = key.shape
        stim = torch.bmm(value.unsqueeze(2), key.unsqueeze(1))  # [B, d, d]
        return stim / (d ** 0.5)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the learned memory recurrence over the token sequence.

        Args:
            tokens: [B, N, d] sequence of patch tokens.
        Returns:
            retrieval: [B, N, d]  the memory retrieval read at each step.
            final_M:   [B, d, d]  final memory state (useful for diagnostics).
        """
        B, N, d = tokens.shape
        device = tokens.device

        # Per-batch memory state, initialized from the learned parameter.
        M = self.M0.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, d, d]

        retrievals = []
        for t in range(N):
            tok = tokens[:, t, :]                    # [B, d]
            # Read: a simple linear-attention-style query over M.
            retrieval = torch.bmm(M, tok.unsqueeze(-1)).squeeze(-1)  # [B, d]
            # Stimulus: rank-1 write from (value=key=tok).
            stim = self._stimulus(tok, tok)          # [B, d, d]
            stim_vec = stim.mean(dim=1)              # [B, d] reduced view for heads

            # Learned surprise: weighted combination of three signals.
            g1 = self.g1(tok)
            g2 = self.g2(retrieval)
            g3 = self.g3(stim_vec)
            # softmax-normalized theta keeps the mixing weights a valid convex combo.
            w = F.softmax(self.theta, dim=0)
            # Surprise is bounded to (0,1) via sigmoid. This is both faithful
            # (Titans' g is bounded) and numerically necessary: the memory
            # update gate alpha = surprise*lr must stay in (0,1) so the
            # recurrence (1-alpha)*M + alpha*stim stays stable across BPTT.
            surprise = torch.sigmoid(
                w[0] * self.surprise_head(g1)
                + w[1] * self.surprise_head(g2)
                + w[2] * self.surprise_head(g3)
            ).squeeze(-1)                            # [B]
            # Learned per-step learning rate in (0, 1).
            lr = torch.sigmoid(self.lr_net(torch.cat([tok, retrieval, stim_vec], dim=-1))).squeeze(-1)  # [B]
            # Update gate = surprise * lr (elementwise over batch).
            alpha = (surprise * lr).view(B, 1, 1)    # [B,1,1]

            # Per-channel forget/decay modulator (learned).
            decay = torch.sigmoid(self.decay_net(tok))  # [B, d]
            decay = decay.view(B, 1, d)

            # L2-style self-update of the memory. Fully differentiable (BPTT).
            M = (1.0 - alpha * decay) * M + alpha * stim

            retrievals.append(retrieval)

        retrieval = torch.stack(retrievals, dim=1)  # [B, N, d]
        return retrieval, M


class MACMixer(nn.Module):
    """Memory-As-Context (MAC) mixer: combines the token stream with the
    neural-memory retrieval plus a small set of persistent memory slots, via
    learned gates. This replaces standard self-attention in a HOPE layer.

    For efficiency the "attention" between tokens and the per-step retrieval is
    realized by a lightweight gated residual rather than an NxN matrix; this
    preserves the MAC spirit (memory is context, not a recurrence target) while
    staying tractable for BPTT over long-ish sequences.
    """

    def __init__(self, d_model: int, d_hidden: int, num_persistent_memory: int = 4):
        super().__init__()
        self.d_model = d_model
        self.memory = NeuralLongTermMemory(d_model, d_hidden)
        self.num_persistent = num_persistent_memory
        # Persistent memory slots: learnable tokens prepended to context.
        self.persistent = nn.Parameter(torch.randn(num_persistent_memory, d_model) * 0.02)

        # Fusion gates.
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.retrieve_gate = nn.Linear(2 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Lightweight FFN applied after fusion.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_model),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, N, d] -> [B, N, d]."""
        B, N, d = tokens.shape
        x = self.norm1(tokens)
        retrieval, _ = self.memory(x)                       # [B, N, d]

        # Token<->persistent-memory interaction via cheap cross-mixing.
        pers = self.persistent.unsqueeze(0).expand(B, -1, -1)  # [B, P, d]
        q = self.q(x)                                         # [B, N, d]
        pers_k = self.k(pers)                                 # [B, P, d]
        pers_v = self.v(pers)                                 # [B, P, d]
        # Softmax over persistent slots gives a cheap "what memory is relevant".
        attn_logits = torch.matmul(q, pers_k.transpose(1, 2)) / (d ** 0.5)  # [B, N, P]
        pers_ctx = torch.matmul(attn_logits.softmax(dim=-1), pers_v)        # [B, N, d]

        # Combine original stream, retrieval, and persistent context with a gate.
        gate = torch.sigmoid(self.retrieve_gate(torch.cat([retrieval, pers_ctx], dim=-1)))
        fused = tokens + gate * (retrieval + pers_ctx)

        # FFN residual.
        out = fused + self.ffn(self.norm2(fused))
        return out
