"""
packages/data
--------------
Market data fetching, caching, and technical indicator computation.

Re-exports the public API surface so consumers can write::

    from data import CCXTMarketDataService, BaseMarketDataService
    from data.indicators import rsi, atr, compute_features
"""

from data.indicators import (
    atr,
    bollinger_bands,
    compute_features,
    donchian_channel,
    ema,
    macd,
    returns,
    rolling_volatility,
    rsi,
    sma,
)
from data.market_data import (
    BaseMarketDataService,
    DataNotAvailableError,
    MarketDataError,
    RateLimitError,
)
from data.services.ccxt_market_data import CCXTMarketDataService

__all__ = [
    # Market data services
    "BaseMarketDataService",
    "CCXTMarketDataService",
    # Market data exceptions
    "MarketDataError",
    "RateLimitError",
    "DataNotAvailableError",
    # Indicators
    "rsi",
    "ema",
    "sma",
    "macd",
    "atr",
    "bollinger_bands",
    "donchian_channel",
    "returns",
    "rolling_volatility",
    "compute_features",
]

# ML features — guarded import (scikit-learn may not be installed)
try:
    from data.ml_features import (
        FEATURE_NAMES,
        build_feature_matrix,
        build_feature_vector_from_bars,
    )

    __all__ += [
        "FEATURE_NAMES",
        "build_feature_matrix",
        "build_feature_vector_from_bars",
    ]
except ImportError:
    pass
