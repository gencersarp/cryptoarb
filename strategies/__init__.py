from .base import BaseStrategy, Signal, Position, Side, Market
from .spot_perp import SpotPerpFundingStrategy
from .perp_perp import PerpPerpDiffStrategy
from .basis_revert import BasisMeanRevertStrategy
try:
    from .statarb import StatArbStrategy
except ModuleNotFoundError:  # optional dependency (statsmodels)
    StatArbStrategy = None

__all__ = [
    "BaseStrategy", "Signal", "Position", "Side", "Market",
    "SpotPerpFundingStrategy",
    "PerpPerpDiffStrategy",
    "BasisMeanRevertStrategy",
    "StatArbStrategy",
]
