"""HOPE-JEPA-SIGReg: a Nested-Learning backbone with per-layer SLOT JEPA
objectives regularized by Sketched Isotropic Gaussian Regularization (LeJEPA)
and slot divergence.

Backbone: a stack of HOPE (Nested-Learning) layers. Each HOPE layer combines:
  * Self-Modifying Titans memory mixer (learned internal optimizer + surprise-
    modulated L2 self-update, full BPTT) -- see `titans.py`.
  * Continuum Memory System: FFN Neural Learning Modules with staggered update
    frequencies -- see `hope.py`.

Each HOPE layer carries a SLOT JEPA predictive head: K shared, diverging slots
each predict the masked-patch embeddings, and the predictions are sparse-mixed
by entmax-1.5 attention (which can output exact zeros) -- see `slots.py`. The
sum over layers is a deep/hierarchical JEPA objective. The same slot set also
assembles the pooled embedding via a sparse slot readout.

SIGReg (`sigreg.py`) -- the regularizer introduced in LeJEPA
(Balestriero & LeCun, 2025, arXiv:2511.08544) -- is the documented fix for the
representation collapse that per-layer self-targeting JEPA would otherwise fall
into. `slot_divergence_loss` (`slots.py`) additionally decorrelates the slots so
each specializes on a different aspect of the data.

`jepa.py` keeps the original non-slot predictor as an ablation baseline.
"""

from .model import HopeJepaModel
from .slots import SlotJEPAPredictor, SlotReadout, slot_divergence_loss
from .entmax import entmax15

__all__ = [
    "HopeJepaModel",
    "SlotJEPAPredictor",
    "SlotReadout",
    "slot_divergence_loss",
    "entmax15",
]
