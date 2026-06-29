"""Tests for the local database and seed loading (offline; no network)."""

import pytest

from asteroid import database


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point the DB at a throwaway directory so tests never touch ~/.asteroid."""
    monkeypatch.setenv("ASTEROID_HOME", str(tmp_path))
    monkeypatch.delenv("ASTEROID_DB", raising=False)
    database.reset_db()
    yield


def test_seed_loads_and_lists():
    bodies = database.list_bodies()
    assert len(bodies) >= 20
    names = {b.name for b in bodies}
    assert any("Ceres" in n for n in names)
    assert any("Apophis" in n for n in names)


def test_lookup_case_insensitive():
    assert database.get_body("apophis") is not None
    assert database.get_body("APOPHIS") is not None
    assert database.get_body("Ceres") is not None


def test_lookup_missing_returns_none():
    assert database.get_body("definitely-not-a-real-object-xyz") is None


def test_body_to_elements_roundtrips_units():
    bennu = database.get_body("Bennu")
    assert bennu is not None
    el = bennu.to_elements()
    # Bennu: a ~ 1.13 AU, e ~ 0.20, NEO/PHA.
    assert el.a == pytest.approx(bennu.a, rel=1e-12)
    assert bennu.neo and bennu.pha


def test_hyperbolic_body_present():
    """At least one interstellar (e>1, a<0) object made it into the seed."""
    bodies = database.list_bodies()
    hyperbolic = [b for b in bodies if b.e >= 1.0]
    assert hyperbolic, "expected a hyperbolic object in the seed"
    assert all(b.a < 0 for b in hyperbolic)


def test_upsert_and_fetch_back():
    custom = database.Body(
        key="testroid", name="Testroid", a=2.0, e=0.1, inc_deg=5.0,
        node_deg=10.0, argp_deg=20.0, M0_deg=30.0, epoch=2451545.0,
        source="test",
    )
    database.upsert_body(custom)
    got = database.get_body("Testroid")
    assert got is not None and got.a == pytest.approx(2.0)
