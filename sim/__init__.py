"""Shared trading simulation for HighRoller.

Lives outside BaselineModels/ on purpose: the Q-learning agent must import the
exact same fee, fill and accounting code as the baselines, or the final
comparison in the paper is meaningless.
"""

from .execution import (
    ACTIONS,
    BUY_DOWN,
    BUY_UP,
    CLOSE,
    ExecutionConfig,
    HOLD,
    Portfolio,
    Side,
    fill_price,
    taker_fee,
)

__all__ = [
    "ACTIONS",
    "BUY_DOWN",
    "BUY_UP",
    "CLOSE",
    "ExecutionConfig",
    "HOLD",
    "Portfolio",
    "Side",
    "fill_price",
    "taker_fee",
]
