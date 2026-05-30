"""Parquet persistence with audit logs.

Every write emits a `persist.write` event with: path, rows, bytes, schema.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import obs


def write_chain(df: pd.DataFrame, data_dir: Path, *, underlying: str,
                expiry: date, snapshot_ts: str) -> Path:
    """Persist an option chain snapshot to Parquet.

    Layout: data_dir/chains/<underlying>/<expiry>.parquet
    Appends to the file if it already exists (multiple snapshots per expiry).
    """
    out_dir = data_dir / "chains" / underlying.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{expiry.isoformat()}.parquet"

    df = df.copy()
    df["_snapshot_ts"] = snapshot_ts

    with obs.timed("persist", "write.chain",
                   underlying=underlying, expiry=str(expiry),
                   path=str(path)) as t:
        if path.exists():
            existing = pq.read_table(path).to_pandas()
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df

        table = pa.Table.from_pandas(combined, preserve_index=False)
        pq.write_table(table, path, compression="snappy")
        size = path.stat().st_size
        t.add(rows=len(df), total_rows=len(combined), bytes=size,
              cols=list(df.columns))

    obs.bump("persist_writes")
    obs.bump("rows_written", by=len(df))
    obs.bump("bytes_written", by=path.stat().st_size)
    return path


def write_bars(df: pd.DataFrame, data_dir: Path, *, ticker: str) -> Path:
    """Persist OHLCV bars to Parquet, one file per ticker."""
    out_dir = data_dir / "underlying"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker.upper()}.parquet"

    with obs.timed("persist", "write.bars",
                   ticker=ticker, path=str(path)) as t:
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path, compression="snappy")
        size = path.stat().st_size
        t.add(rows=len(df), bytes=size, cols=list(df.columns))

    obs.bump("persist_writes")
    obs.bump("rows_written", by=len(df))
    obs.bump("bytes_written", by=path.stat().st_size)
    return path
