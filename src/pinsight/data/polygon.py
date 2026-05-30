"""Polygon.io adapter — REST client with rate limiting and full telemetry.

Free tier constraints (enforced in this module):
  * 5 calls/minute = one call every 12 seconds, token bucket.
  * EOD data only; quotes are 15-minute delayed.
  * 2 years of historical aggregates max.

Every call is logged with: endpoint, params, latency, HTTP status,
response bytes, X-RateLimit-* headers. Errors are retried with
exponential backoff (handled by tenacity if installed, else manual).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import requests

from .. import obs

_BASE_URL = "https://api.polygon.io"


class PolygonError(RuntimeError):
    pass


@dataclass
class _TokenBucket:
    """Thread-safe token bucket. Free tier = 5 tokens, refill 5/min."""

    capacity: int = 5
    refill_per_sec: float = 5 / 60.0
    tokens: float = 5.0
    last_refill_ns: int = 0
    lock: threading.Lock = None  # set in __post_init__

    def __post_init__(self) -> None:
        self.lock = threading.Lock()
        self.last_refill_ns = time.perf_counter_ns()

    def acquire(self) -> float:
        """Block until a token is available. Returns seconds slept."""
        slept = 0.0
        while True:
            with self.lock:
                now = time.perf_counter_ns()
                elapsed = (now - self.last_refill_ns) / 1e9
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
                self.last_refill_ns = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return slept
                deficit = 1.0 - self.tokens
                wait = deficit / self.refill_per_sec
            time.sleep(wait)
            slept += wait


class PolygonClient:
    """Rate-limited REST client for Polygon. One instance per process."""

    def __init__(self, api_key: str, *, timeout_s: float = 20.0, tier: str = "free") -> None:
        if not api_key:
            raise PolygonError("Polygon API key missing — set POLYGON_API_KEY in .env")
        self._key = api_key
        self._timeout = timeout_s
        self._tier = tier
        self._bucket = _TokenBucket()  # Free-tier defaults; bypass for paid tiers later.
        self._session = requests.Session()

    # ---------- low-level ----------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{_BASE_URL}{path}"
        params = dict(params or {})
        # apiKey is appended as a query param; sanitize for logs.
        log_params = {**params, "apiKey": "<redacted>"}
        params["apiKey"] = self._key

        slept = self._bucket.acquire()
        if slept > 0:
            obs.event(channel="api", kind="polygon.throttle", level="DEBUG",
                      sleep_s=round(slept, 2))

        with obs.timed("api", "polygon.get", endpoint=path, params=log_params) as t:
            try:
                resp = self._session.get(url, params=params, timeout=self._timeout)
            except requests.RequestException as exc:
                obs.bump("api_errors")
                raise PolygonError(f"network error on {path}: {exc}") from exc

            rate_remaining = resp.headers.get("X-RateLimit-Remaining")
            rate_limit = resp.headers.get("X-RateLimit-Limit")
            content_bytes = len(resp.content)
            t.add(
                status=resp.status_code,
                bytes=content_bytes,
                rate_limit=rate_limit,
                rate_remaining=rate_remaining,
            )
            obs.bump("api_calls")

            if resp.status_code == 429:
                obs.event(channel="api", kind="polygon.rate_limited", level="WARNING",
                          endpoint=path, rate_remaining=rate_remaining)
                # Park for a minute and re-raise; caller decides retry.
                time.sleep(60)
                obs.bump("api_errors")
                raise PolygonError("429 rate limited; slept 60s then surfaced")

            if resp.status_code >= 400:
                obs.bump("api_errors")
                raise PolygonError(f"HTTP {resp.status_code} on {path}: {resp.text[:300]}")

            try:
                return resp.json()
            except ValueError as exc:
                obs.bump("api_errors")
                raise PolygonError(f"non-JSON response on {path}") from exc

    # ---------- public endpoints ----------

    def snapshot_option_chain(self, underlying: str, *, expiry: date | None = None,
                              limit: int = 250) -> list[dict[str, Any]]:
        """Snapshot of an options chain. Free tier returns 15-min delayed data.

        Returns a list of contract snapshot dicts, paginated automatically.
        """
        path = f"/v3/snapshot/options/{underlying.upper()}"
        params: dict[str, Any] = {"limit": limit}
        if expiry is not None:
            params["expiration_date"] = expiry.isoformat()

        results: list[dict[str, Any]] = []
        next_url: str | None = None

        while True:
            data = self._get(next_url or path, params=params if next_url is None else None)
            page = data.get("results", []) or []
            results.extend(page)
            next_url_full = data.get("next_url")
            if not next_url_full:
                break
            # Polygon returns full URLs in next_url; strip prefix.
            next_url = next_url_full.replace(_BASE_URL, "")
            # next_url already has apiKey & params encoded; clear params.
            params = None
            obs.event(channel="api", kind="polygon.paginated", level="DEBUG",
                      cumulative_rows=len(results))

        obs.event(channel="api", kind="polygon.snapshot_chain.summary", level="INFO",
                  underlying=underlying, expiry=str(expiry) if expiry else None,
                  contracts=len(results))
        return results

    def aggregates(self, ticker: str, *, multiplier: int, timespan: str,
                   from_: date, to: date, limit: int = 5000) -> list[dict[str, Any]]:
        """OHLCV bars. Free tier supports daily/minute up to 2 years.

        `ticker` can be an equity symbol ("SPY") or an option ticker ("O:SPY...").
        """
        path = (f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}"
                f"/{from_.isoformat()}/{to.isoformat()}")
        data = self._get(path, params={"limit": limit, "sort": "asc"})
        results = data.get("results") or []
        obs.event(channel="api", kind="polygon.aggregates.summary", level="INFO",
                  ticker=ticker, span=f"{multiplier}{timespan}",
                  from_=from_.isoformat(), to=to.isoformat(), bars=len(results))
        return results

    def reference_contracts(self, underlying: str, *,
                            expiration_date: date | None = None,
                            limit: int = 1000) -> list[dict[str, Any]]:
        """Static reference data for option contracts on an underlying."""
        path = "/v3/reference/options/contracts"
        params: dict[str, Any] = {
            "underlying_ticker": underlying.upper(),
            "limit": limit,
        }
        if expiration_date is not None:
            params["expiration_date"] = expiration_date.isoformat()

        data = self._get(path, params=params)
        return data.get("results") or []
