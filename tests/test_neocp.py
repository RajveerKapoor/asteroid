"""Tests for the NEO Confirmation Page ('open problems') layer.

Offline tests cover NEOCP JSON parsing and the CLI paths (list + determine,
with the network monkeypatched) — including the genuine end-to-end act of
solving a preliminary orbit from raw observations. Live network calls run only
when ASTEROID_LIVE_TESTS=1 is set.
"""

import os

import pytest

from asteroid import cli, database, fetch, frames, iod
from asteroid.bodies import earth_position
from asteroid.propagate import OrbitalElements, state_vector


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTEROID_HOME", str(tmp_path))
    monkeypatch.delenv("ASTEROID_DB", raising=False)
    database.reset_db()
    yield


# --------------------------------------------------------------------------- #
# Parsing & heuristics (offline)
# --------------------------------------------------------------------------- #
_NEOCP_ROW = {
    "Temp_Desig": "CEQCZT2", "Score": 98, "Discovery_year": 2026,
    "Discovery_month": 6, "Discovery_day": 27.4, "R.A.": 23.11, "Decl.": 8.9,
    "V": 21.1, "NObs": 23, "Arc": 2.01, "H": 20.7, "Not_Seen_dys": 0.43,
}


def test_parse_neocp_row():
    o = fetch._parse_neocp_row(_NEOCP_ROW)
    assert o.designation == "CEQCZT2"
    assert o.score == 98
    assert o.discovery == "2026-06-27"
    assert o.n_obs == 23
    assert o.arc_days == pytest.approx(2.01)
    assert o.ra_deg == pytest.approx(23.11 * 15.0)     # hours -> degrees
    assert o.name == "CEQCZT2"


def test_solvable_heuristic():
    assert fetch._parse_neocp_row(_NEOCP_ROW).solvable               # 23 obs, 2 d
    short = dict(_NEOCP_ROW, NObs=2, Arc=0.0)
    assert not fetch._parse_neocp_row(short).solvable


# --------------------------------------------------------------------------- #
# CLI list (network monkeypatched)
# --------------------------------------------------------------------------- #
def test_neocp_list_renders(capsys, monkeypatch):
    fake = [
        fetch.NeocpObject("P22nJzF", 100, "2026-06-22", 100.0, 5.0, 21.6, 18,
                          12.0, 9.5, 0.3),
        fetch.NeocpObject("A11E5qG", 100, "2026-06-26", 40.0, -23.0, 22.0, 3,
                          0.01, 29.1, 3.7),
    ]
    monkeypatch.setattr(fetch, "fetch_neocp_list", lambda *a, **k: fake)
    assert cli.main(["--neocp"]) == 0
    out = capsys.readouterr().out
    assert "Confirmation Page" in out
    assert "P22nJzF" in out
    assert "too short" in out          # the 0.01-day arc is flagged unsolvable


# --------------------------------------------------------------------------- #
# CLI determine: solve a preliminary orbit from raw observations
# --------------------------------------------------------------------------- #
def _synthetic_observations(truth, days):
    obs = []
    for d in days:
        jd = truth.epoch + d
        geo = state_vector(truth, jd)[0] - earth_position(jd)
        ra, dec = frames.equatorial_to_radec(frames.ecliptic_to_equatorial(geo))
        obs.append(iod.Observation(jd=jd, ra_deg=ra % 360, dec_deg=dec, observer=None))
    return obs


def test_neocp_determine_flags_preliminary(capsys, monkeypatch):
    """A designation only on NEOCP falls back from the archive and solves,
    clearly labelled as a short-arc preliminary orbit."""
    truth = OrbitalElements.from_degrees(
        a=2.3, e=0.35, inc_deg=8.0, node_deg=100.0, argp_deg=50.0,
        M0_deg=12.0, epoch=2461210.5)
    obs = _synthetic_observations(truth, (0, 1, 2, 3, 4, 5, 6))

    def not_in_archive(*a, **k):
        raise fetch.FetchError("MPC has no observations")
    monkeypatch.setattr(fetch, "fetch_mpc_observations", not_in_archive)
    monkeypatch.setattr(fetch, "fetch_neocp_observations", lambda *a, **k: obs)

    assert cli.main(["P22nJzF", "--determine"]) == 0
    out = capsys.readouterr().out
    assert "Preliminary orbit solved" in out
    assert "preliminary" in out.lower()
    # The solved object is saved and usable like any other body.
    assert database.get_body("P22nJzF") is not None


def test_neocp_shorthand_routes_to_determine(capsys, monkeypatch):
    truth = OrbitalElements.from_degrees(
        a=1.8, e=0.5, inc_deg=4.0, node_deg=120.0, argp_deg=70.0,
        M0_deg=350.0, epoch=2461210.5)
    obs = _synthetic_observations(truth, (0, 1.5, 3, 4.5, 6, 7.5))
    monkeypatch.setattr(fetch, "fetch_mpc_observations",
                        lambda *a, **k: (_ for _ in ()).throw(fetch.FetchError("x")))
    monkeypatch.setattr(fetch, "fetch_neocp_observations", lambda *a, **k: obs)
    # `<desig> --neocp` is shorthand for determining a confirmation-page object.
    assert cli.main(["X9test", "--neocp"]) == 0
    assert "Preliminary orbit solved" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Live (network-gated)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.environ.get("ASTEROID_LIVE_TESTS") != "1",
                    reason="set ASTEROID_LIVE_TESTS=1 to run network tests")
def test_live_neocp_fetch_and_solve():
    objs = fetch.fetch_neocp_list()
    if not objs:
        pytest.skip("NEO Confirmation Page is empty right now")
    target = next((o for o in objs if o.solvable), objs[0])
    observations = fetch.fetch_neocp_observations(target.designation)
    assert len(observations) >= 3
    # If the arc is long enough, a determination should converge to a finite orbit.
    if target.arc_days and target.arc_days >= 1.0:
        sol = iod.determine_orbit(observations, name=target.designation)
        assert sol.rms_arcsec < 10.0
