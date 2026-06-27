"""Low-memory optimizers for HOPE-JEPA LLM training.

  * ``LOMO``     -- LOw-Memory Optimization: fused SGD with global grad clip
                    (two backward passes; faithful to Lv et al. 2023).
  * ``AdaLomo``  -- momentum + per-tensor clip, single backward pass (the
                    fast/recommended LOMO-family variant; OpenLMLab AdaLomo).
  * ``LISA``     -- Layerwise Importance Sampled AdamW (Pan et al., NeurIPS
                    2024): activate a random K of the base decoder layers per
                    step, freeze the rest (no state, pruned backward).

See each module's docstring for the exact usage / training-loop contract.
"""

from .lomo import LOMO, AdaLomo, _FusedOptimizerBase
from .lisa import LISA

__all__ = ["LOMO", "AdaLomo", "LISA"]
