"""Data ingestion adapters.

Each provider lives in its own module and exposes the same interface:

    fetch_chain(symbol: str, expiry: date) -> pd.DataFrame
    fetch_underlying(symbol: str, start, end, interval) -> pd.DataFrame

The orchestration layer (M1) decides which provider to call based on config
and provider availability.
"""
