"""Currency conversion, validation and stale-rate fallback tests."""
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
import urllib.error

import pytest

from sustainfab.currency import (
    REFERENCE_DATE, REFERENCE_RATES_PER_USD,
    ExchangeRates, format_money, get_usd_exchange_rates, tariff_in_currency,
)


def reference_rates():
    return ExchangeRates(dict(REFERENCE_RATES_PER_USD), REFERENCE_DATE, "WAMA", "wama", True)


@pytest.mark.parametrize(("currency", "expected"), [
    ("USD", "$64.14"), ("EUR", "€64.14"),
    ("GBP", "£64.14"), ("NGN", "₦64.14"),
])
def test_currency_symbols(currency, expected):
    assert format_money(64.14, currency) == expected


def test_usd_tariff_currency_conversion():
    rates = reference_rates()
    assert tariff_in_currency(0.15, "USD", rates) == pytest.approx(0.15)
    assert tariff_in_currency(0.15, "NGN", rates) == pytest.approx(203.307225)
    assert tariff_in_currency(0.15, "EUR", rates) == pytest.approx(0.13572255)
    assert tariff_in_currency(0.15, "GBP", rates) == pytest.approx(0.1133493)
    assert format_money(427.6 * tariff_in_currency(0.15, "NGN", rates), "NGN").startswith("₦")


def test_bad_currency_and_tariff_are_rejected():
    with pytest.raises(ValueError):
        tariff_in_currency(0.15, "CAD", reference_rates())
    with pytest.raises(ValueError):
        tariff_in_currency(-0.1, "NGN", reference_rates())
    with pytest.raises(ValueError):
        format_money(10, "CAD")


def test_fresh_provider_response(monkeypatch):
    payload = {
        "result": "success", "base_code": "USD",
        "time_last_update_unix": int(datetime.now(timezone.utc).timestamp()),
        "rates": dict(REFERENCE_RATES_PER_USD),
    }
    monkeypatch.setattr(
        "sustainfab.currency.urllib.request.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(json.dumps(payload).encode()),
    )
    result = get_usd_exchange_rates()
    assert not result.fallback
    assert result.provider.startswith("ExchangeRate-API")
    assert result.per_usd["NGN"] == pytest.approx(1355.3815)


def test_stale_provider_response_is_not_treated_as_live(monkeypatch):
    payload = {
        "result": "success", "base_code": "USD",
        "time_last_update_unix": 1000000000,
        "rates": dict(REFERENCE_RATES_PER_USD),
    }
    monkeypatch.setattr(
        "sustainfab.currency.urllib.request.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(json.dumps(payload).encode()),
    )
    result = get_usd_exchange_rates()
    assert result.fallback
    assert result.as_of == "2026-11-16"


def test_network_failure_falls_back_with_date(monkeypatch):
    def offline(*_args, **_kwargs):
        raise urllib.error.URLError("Offline")
    monkeypatch.setattr("sustainfab.currency.urllib.request.urlopen", offline)
    result = get_usd_exchange_rates()
    assert result.fallback
    assert result.as_of == REFERENCE_DATE
    assert result.per_usd == REFERENCE_RATES_PER_USD


def test_investment_amounts_round_trip_across_currencies():
    """Changing selected currency must not relabel or inflate the USD amount."""
    from sustainfab.currency import usd_to_selected, selected_to_usd
    rates = reference_rates()
    for currency in ("USD", "NGN", "EUR", "GBP"):
        shown = usd_to_selected(12000.0, currency, rates)
        assert selected_to_usd(shown, currency, rates) == pytest.approx(12000.0)
        assert shown == pytest.approx(12000.0 * rates.per_usd[currency])


def test_cross_currency_conversion_preserves_economic_amount():
    from sustainfab.currency import usd_to_selected, selected_to_usd
    rates = reference_rates()
    naira_input = usd_to_selected(500.0, "NGN", rates)
    canonical = selected_to_usd(naira_input, "NGN", rates)
    euros = usd_to_selected(canonical, "EUR", rates)
    assert euros == pytest.approx(500.0 * rates.per_usd["EUR"])


def test_investment_rejects_invalid_fx_and_negative_inputs():
    from sustainfab.currency import usd_to_selected, selected_to_usd
    rates = reference_rates()
    with pytest.raises(ValueError):
        usd_to_selected(-5.0, "NGN", rates)
    with pytest.raises(ValueError):
        selected_to_usd(100.0, "XYZ", rates)
    bad_rates = replace(rates, per_usd={**rates.per_usd, "NGN": 0.0})
    with pytest.raises(ValueError):
        selected_to_usd(100.0, "NGN", bad_rates)
