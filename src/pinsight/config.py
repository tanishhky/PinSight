"""Runtime configuration loaded from environment.

Reads `.env` if present (via python-dotenv) and exposes a typed Config object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    data_dir: Path
    log_dir: Path
    log_level: str
    underlying: str
    rfr_source: str
    rfr_manual: float
    tradier_api_key: str | None
    tradier_env: str
    polygon_api_key: str | None
    alpaca_api_key: str | None
    alpaca_api_secret: str | None
    alpaca_env: str
    fred_api_key: str | None


def load() -> Config:
    return Config(
        data_dir=Path(os.getenv("PINSIGHT_DATA_DIR", "./data")).resolve(),
        log_dir=Path(os.getenv("PINSIGHT_LOG_DIR", "./logs")).resolve(),
        log_level=os.getenv("PINSIGHT_LOG_LEVEL", "INFO"),
        underlying=os.getenv("PINSIGHT_UNDERLYING", "SPY"),
        rfr_source=os.getenv("PINSIGHT_RFR_SOURCE", "fred"),
        rfr_manual=float(os.getenv("PINSIGHT_RFR_MANUAL", "0.05")),
        tradier_api_key=os.getenv("TRADIER_API_KEY") or None,
        tradier_env=os.getenv("TRADIER_ENV", "sandbox"),
        polygon_api_key=os.getenv("POLYGON_API_KEY") or None,
        alpaca_api_key=os.getenv("ALPACA_API_KEY") or None,
        alpaca_api_secret=os.getenv("ALPACA_API_SECRET") or None,
        alpaca_env=os.getenv("ALPACA_ENV", "paper"),
        fred_api_key=os.getenv("FRED_API_KEY") or None,
    )
