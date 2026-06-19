"""HOPE-JEPA-SIGReg: a Nested-Learning backbone with per-layer JEPA objectives
regularized by Sketched Isotropic Gaussian Regularization (LeJEPA).

Backbone: a stack of HOPE (Nested-Learning) layers. Each HOPE layer combines:
  * Self-Modifying Titans memory mixer (learned internal optimizer + surprise-
    modulated L2 self-update, full BPTT) -- see `titans.py`.
  * Continuum Memory System: FFN Neural Learning Modules with staggered update
    frequencies -- see `hope.py`.

Each HOPE layer carries a JEPA predictive head that predicts the masked-patch
embeddings from the context-patch embeddings in latent space (`jepa.py`). The
sum over layers is a deep/hierarchical JEPA objective.

SIGReg (`sigreg.py`) -- the regularizer introduced in LeJEPA
(Balestriero & LeCun, 2025, arXiv:2511.08544) -- is the documented fix for the
representation collapse that per-layer self-targeting JEPA would otherwise fall
into.
"""

from .model import HopeJepaModel

__all__ = ["HopeJepaModel"]
