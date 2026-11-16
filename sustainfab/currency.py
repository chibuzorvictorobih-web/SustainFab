"""Transparent reference FX conversions for SustainFab electricity-only costs.

Tariff inputs are USD/kWh; select a display currency separately. Rates are
indicative, never tradeable quotes. An offline fallback is explicitly dated.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import urllib.request


SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "NGN": "₦"}
REFERENCE_DATE = "2026-11-16"
REFERENCE_SOURCE = "West African Monetary Agency (WAMA) daily rates"
REFERENCE_URL = (
    "https://exrates.amao-wama.org/daily-rates"
    "?start=2026-11-16&end=2026-11-16&format=pdf"
)
LIVE_URL = "https://open.er-api.com/v6/latest/USD"
LIVE_ATTRIBUTION = "https://www.exchangerate-api.com"
# Row-currency units per 1 USD, 16 November 2026 (WAMA daily rates).
REFERENCE_RATES_PER_USD = {
    "USD": 1.0, "EUR": 0.904817, "GBP": 0.755662, "NGN": 1355.381500,
}


@dataclass(frozen=True)
class ExchangeRates:
    per_usd: dict[str, float]
    as_of: str
    provider: str
    source_url: str
    fallback: bool


def _validated_rates(payload: dict) -> ExchangeRates:
    if payload.get("result") != "success" or payload.get("base_code") != "USD":
        raise ValueError("FX provider returned a non-USD or unsuccessful response.")
    raw = payload.get("rates", {})
    rates: dict[str, float] = {}
    for code in SYMBOLS:
        rate = float(raw[code])
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("FX provider returned a nonpositive or invalid rate.")
        rates[code] = rate
    if abs(rates["USD"] - 1.0) > 1e-8:
        raise ValueError("FX provider did not return a USD-based table.")

    timestamp = int(payload["time_last_update_unix"])
    updated = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    age = datetime.now(timezone.utc) - updated
    # Do not quietly treat a stale cached FX response as current.
    if age < -timedelta(hours=2) or age > timedelta(days=3):
        raise ValueError("FX data timestamp is not recent.")
    return ExchangeRates(
        per_usd=rates, as_of=updated.strftime("%Y-%m-%d %H:%M UTC"),
        provider="ExchangeRate-API (daily reference)",
        source_url=LIVE_ATTRIBUTION, fallback=False,
    )


def get_usd_exchange_rates() -> ExchangeRates:
    """Fetch daily reference FX. Fall back to a conspicuously dated snapshot."""
    try:
        request = urllib.request.Request(
            LIVE_URL, headers={"User-Agent": "SustainFab/1.2 FX reference dashboard"}
        )
        with urllib.request.urlopen(request, timeout=7) as response:
            payload = json.loads(response.read(300000))
        return _validated_rates(payload)
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return ExchangeRates(
            per_usd=dict(REFERENCE_RATES_PER_USD),
            as_of=REFERENCE_DATE,
            provider=REFERENCE_SOURCE,
            source_url=REFERENCE_URL,
            fallback=True,
        )


def tariff_in_currency(usd_tariff_per_kwh: float, quote: str, rates: ExchangeRates) -> float:
    """Convert a USD/kWh tariff once; the simulation handles kWh multiplication."""
    if quote not in SYMBOLS:
        raise ValueError("Unsupported currency.")
    if not math.isfinite(usd_tariff_per_kwh) or usd_tariff_per_kwh < 0:
        raise ValueError("Tariff must be a non-negative, finite number.")
    return usd_tariff_per_kwh * rates.per_usd[quote]


def format_money(amount: float, currency: str, digits: int = 2) -> str:
    if currency not in SYMBOLS:
        raise ValueError("Unsupported currency.")
    if not math.isfinite(amount):
        raise ValueError("Amount must be finite.")
    if digits not in (2, 4):
        raise ValueError("Only two or four decimal places are supported.")
    return f"{SYMBOLS[currency]}{amount:,.{digits}f}"
