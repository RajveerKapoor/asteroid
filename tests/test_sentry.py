"""Tests for the NASA Sentry impact-risk ('open problems') layer.

Offline tests cover parsing of the Sentry JSON shapes and the CLI rendering
(with the network monkeypatched). The live network calls run only when
ASTEROID_LIVE_TESTS=1 is set.
"""

import os

import pytest

from asteroid import cli, database, fetch


# --------------------------------------------------------------------------- #
# Parsing & helpers (offline)
# --------------------------------------------------------------------------- #
_LIST_ROW = {
    "des": "2000 SG344", "fullname": "(2000 SG344)", "ip": "2.74e-03",
    "ps_cum": "-2.77", "ps_max": "-3.11", "ts_max": "0", "diameter": "0.037",
    "h": "24.8", "v_inf": "1.358", "n_imp": 100, "range": "2069-2122",
    "last_obs": "2000-10-03",
}


def test_parse_sentry_row():
    r = fetch._parse_sentry_row(_LIST_ROW)
    assert r.designation == "2000 SG344"
    assert r.impact_prob == pytest.approx(2.74e-03)
    assert r.palermo_cum == pytest.approx(-2.77)
    assert r.torino_max == 0
    assert r.diameter_km == pytest.approx(0.037)
    assert r.year_range == "2069-2122"
    assert r.name == "2000 SG344"


def test_parse_sentry_row_tolerates_missing_fields():
    r = fetch._parse_sentry_row({"des": "X"})
    assert r.designation == "X"
    assert r.impact_prob == 0.0
    assert r.diameter_km is None
    assert r.torino_max == 0


def test_year_range_from_impacts():
    impacts = [{"date": "2069-10-01.5"}, {"date": "2122-03-11.2"},
               {"date": "2095-01-01.0"}]
    assert fetch._year_range_from_impacts(impacts) == "2069-2122"
    assert fetch._year_range_from_impacts([{"date": "2050-01-01"}]) == "2050"
    assert fetch._year_range_from_impacts([]) == ""


def test_impact_odds_and_palermo_style():
    assert cli._impact_odds(1e-3) == "1 in 1,000"
    assert cli._impact_odds(0.0) == "—"
    assert cli._palermo_style(0.5) == "bold red"      # >0: genuinely concerning
    assert cli._palermo_style(-1.0) == "yellow"       # -2..0: notable
    assert cli._palermo_style(-5.0) == "white"        # low


# --------------------------------------------------------------------------- #
# CLI rendering (network monkeypatched)
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTEROID_HOME", str(tmp_path))
    monkeypatch.delenv("ASTEROID_DB", raising=False)
    database.reset_db()
    yield


def test_risk_list_renders(capsys, monkeypatch):
    fake = [
        fetch.SentryRisk("2000 SG344", "(2000 SG344)", 2.74e-3, -2.77, -3.11, 0,
                         0.037, 24.8, 1.36, 100, "2069-2122", "2000-10-03"),
        fetch.SentryRisk("99942", "(99942) Apophis", 0.0, -3.0, -3.2, 0,
                         0.34, 19.7, 5.8, 1, "2068", "2021-03-10"),
    ]
    monkeypatch.setattr(fetch, "fetch_sentry_list", lambda *a, **k: fake)
    assert cli.main(["--risk-list", "5"]) == 0
    out = capsys.readouterr().out
    assert "impact-risk watchlist" in out
    assert "2000 SG344" in out
    assert "1 in 365" in out                 # 1/2.74e-3 ≈ 365


def test_risk_overlay_with_risk(capsys, monkeypatch):
    detail = fetch.SentryObject("101955", "101955 Bennu", 1.0e-3, -1.40, 0,
                                0.49, 12.68, 1421.0, 157, "2178-2290", 554, 7693.0)
    monkeypatch.setattr(fetch, "fetch_sentry_object", lambda *a, **k: detail)
    assert cli.main(["Bennu", "--risk", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "Sentry risk" in out
    assert "1 in 1,000" in out
    assert "megatons" in out


def test_risk_overlay_no_risk(capsys, monkeypatch):
    monkeypatch.setattr(fetch, "fetch_sentry_object", lambda *a, **k: None)
    assert cli.main(["Ceres", "--risk", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "no known Earth-impact risk" in out.lower() or "not on NASA" in out


# --------------------------------------------------------------------------- #
# Live (network-gated)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.environ.get("ASTEROID_LIVE_TESTS") != "1",
                    reason="set ASTEROID_LIVE_TESTS=1 to run network tests")
def test_live_sentry_list_and_object():
    risks = fetch.fetch_sentry_list()
    assert len(risks) > 100
    # Sorted most-hazardous first.
    assert risks[0].palermo_cum >= risks[-1].palermo_cum
    # A known Sentry object resolves; a safe one returns None.
    bennu = fetch.fetch_sentry_object("101955")
    assert bennu is not None and bennu.impact_prob > 0
    assert fetch.fetch_sentry_object("1") is None     # (1) Ceres: no risk
