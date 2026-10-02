"""Controller mechanics for future Ugi synthesis-guided sampling.

The functions in this module are qualified only with diagnostic fake values.
They do not define a synthesis score and do not authorize production guidance.
At guidance strength zero, ancestry is always the identity map so the controller
cannot perturb the frozen generator. At nonzero guidance, an exactly uniform
effective ancestry law also preserves identity rather than adding sampling noise.
"""

from __future__ import annotations
