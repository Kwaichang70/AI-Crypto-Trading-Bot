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

# ML features  -- guarded import (scikit-learn may not be installed)
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

# Sentiment / Fear & Greed Index - guarded import (aiohttp may not be installed)
try:
    from data.sentiment import (
        FearGreedClient,
        FearGreedSnapshot,
        get_global_client,
        set_global_client,
    )

    __all__ += [
        "FearGreedClient",
        "FearGreedSnapshot",
        "get_global_client",
        "set_global_client",
    ]
except ImportError:
    pass

# CoinGecko market signals - guarded import (aiohttp may not be installed)
try:
    from data.market_signals import (
        CoinGeckoClient,
        CoinGeckoSnapshot,
        get_global_client as get_coingecko_client,
        set_global_client as set_coingecko_client,
    )

    __all__ += [
        "CoinGeckoClient",
        "CoinGeckoSnapshot",
        "get_coingecko_client",
        "set_coingecko_client",
    ]
except ImportError:
    pass

# FRED macro data - guarded import (aiohttp may not be installed)
try:
    from data.macro_data import (
        FREDClient,
        MacroSnapshot,
        get_global_client as get_fred_client,
        set_global_client as set_fred_client,
    )

    __all__ += [
        "FREDClient",
        "MacroSnapshot",
        "get_fred_client",
        "set_fred_client",
    ]
except ImportError:
    pass

# Whale Alert on-chain tracker - guarded import (aiohttp may not be installed)
try:
    from data.whale_tracker import (
        WhaleAlertClient,
        WhaleFlowSnapshot,
        get_global_client as get_whale_client,
        set_global_client as set_whale_client,
    )

    __all__ += [
        "WhaleAlertClient",
        "WhaleFlowSnapshot",
        "get_whale_client",
        "set_whale_client",
    ]
except ImportError:
    pass
