"""Offline tests for the online-OD machinery: parsing, topocentry, windowing.

The network call itself is exercised only when ASTEROID_LIVE_TESTS=1 is set.
"""

import os

import numpy as np
import pytest

from asteroid import fetch, frames, iod
from asteroid.bodies import geocenter_position, earth_position, moon_position_geocentric
from asteroid.constants import EARTH_RADIUS_AU, AU_KM


# --------------------------------------------------------------------------- #
# Time & geometry
# --------------------------------------------------------------------------- #
def test_gmst_at_j2000():
    # Greenwich mean sidereal time at J2000.0 is ~280.46 deg.
    assert frames.gmst_deg(2451545.0) == pytest.approx(280.4606, abs=1e-3)


def test_gmst_advances_about_360_99_deg_per_day():
    g0 = frames.gmst_deg(2451545.0)
    g1 = frames.gmst_deg(2451546.0)
    assert (g1 - g0) % 360.0 == pytest.approx(360.98564736629 % 360.0, abs=1e-3)


def test_geocenter_offset_is_earth_moon_barycentre():
    """Earth's centre is ~4,700 km from the Standish (barycentre) position."""
    for date in ["2024-01-08", "2024-06-15", "2026-06-29"]:
        jd = frames.parse_date(date)
        off_km = np.linalg.norm(earth_position(jd) - geocenter_position(jd)) * AU_KM
        assert 3000.0 < off_km < 6000.0


def test_moon_distance_range():
    dists = [np.linalg.norm(moon_position_geocentric(frames.parse_date(d))) * AU_KM
             for d in ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"]]
    assert min(dists) > 350_000 and max(dists) < 410_000


def test_topocentric_observer_near_geocentre():
    """A ground observatory sits ~1 Earth radius from the geocentre."""
    obscodes = {"691": (248.39966, 0.849466, 0.526479)}
    jd = frames.parse_date("2021-03-16 08:00")
    obs = iod.observer_position_topocentric("691", jd, obscodes)
    offset = np.linalg.norm(obs - geocenter_position(jd))
    assert 0.95 * EARTH_RADIUS_AU < offset < 1.01 * EARTH_RADIUS_AU


def test_unknown_station_falls_back_to_geocentre():
    jd = frames.parse_date("2021-03-16")
    obs = iod.observer_position_topocentric("C51", jd, {})  # space telescope
    assert np.allclose(obs, geocenter_position(jd))


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
OBSCODE_SAMPLE = (
    "Code  Long.   cos      sin    Name\n"
    "500   0.000000.000000+0.000000Geocentric\n"
    "691 248.399660.849466+0.526479Steward Observatory, Kitt Peak-Spacewatch\n"
    "250                           Hubble Space Telescope\n"
    "F51 203.744090.936241+0.351543Pan-STARRS 1, Haleakala\n"
)


def test_parse_obscodes():
    codes = fetch.parse_obscodes(OBSCODE_SAMPLE)
    assert codes["691"] == pytest.approx((248.39966, 0.849466, 0.526479))
    assert codes["F51"][0] == pytest.approx(203.74409)
    assert codes["500"] == pytest.approx((0.0, 0.0, 0.0))
    assert "250" not in codes  # space telescope: no parallax constants


def test_parse_obs80_line():
    line = ("99942K04M04N  C2004 03 15.10789 04 06 08.08 "
            "+16 55 04.6                om6394691")
    jd, ra, dec, stn = fetch._parse_obs80_line(line)
    assert stn == "691"
    assert ra == pytest.approx((4 + 6/60 + 8.08/3600) * 15.0, abs=1e-6)
    assert dec == pytest.approx(16 + 55/60 + 4.6/3600, abs=1e-6)
    assert frames.format_jd(jd, with_time=False) == "2004-03-15"


def test_parse_obs80_skips_satellite_second_line():
    # A satellite second line ('s' in column 15) carries no optical RA/Dec.
    line = "     K14A00A  s2014 01 01.00000 1 - 0.1234 + 0.5678 - 0.9012   ~abcdC51"
    assert fetch._parse_obs80_line(line) is None


# --------------------------------------------------------------------------- #
# Window selection
# --------------------------------------------------------------------------- #
def _obs(jd):
    return iod.Observation(jd=jd, ra_deg=10.0, dec_deg=5.0)


def test_candidate_windows_picks_dense_cluster():
    base = frames.parse_date("2010-01-01")
    sparse = [_obs(base + 400 * k) for k in range(6)]          # one every ~13 months
    dense = [_obs(base + 2000 + d) for d in range(0, 40, 2)]   # 20 nights in 40 days
    windows = iod.candidate_windows(sparse + dense, window_days=45, n_windows=4)
    assert windows
    best = max(windows, key=len)
    span = best[-1].jd - best[0].jd
    assert len(best) >= 15 and span <= 45.0


def test_short_arc_determine_unchanged():
    """A short arc bypasses windowing (regression guard for synthetic OD)."""
    from asteroid.propagate import OrbitalElements, state_vector
    from asteroid.bodies import earth_position as ep
    truth = OrbitalElements.from_degrees(
        a=0.9223592206975018, e=0.1911492279663492, inc_deg=3.340996879880978,
        node_deg=203.8936514240762, argp_deg=126.6795706895841,
        M0_deg=175.3304026592739, epoch=2461200.5, n_deg=1.112638115271892)
    base = frames.parse_date("2027-03-01")
    obs = []
    for d in (0, 6, 12, 18, 24, 30):
        jd = base + d
        ra, dec = frames.equatorial_to_radec(
            frames.ecliptic_to_equatorial(state_vector(truth, jd)[0] - ep(jd)))
        obs.append(iod.Observation(jd=jd, ra_deg=ra % 360, dec_deg=dec))
    sol = iod.determine_orbit(obs, light_time=False)
    assert sol.rms_arcsec < 1e-2
    assert sol.elements.a == pytest.approx(truth.a, abs=1e-4)


# --------------------------------------------------------------------------- #
# Live (opt-in) end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.environ.get("ASTEROID_LIVE_TESTS") != "1",
                    reason="set ASTEROID_LIVE_TESTS=1 to run network tests")
def test_live_mpc_determination():
    obs = fetch.fetch_mpc_observations("99942")
    sol = iod.determine_orbit(obs)
    assert sol.rms_arcsec < 3.0
    assert sol.elements.a == pytest.approx(0.9224, abs=0.01)


@pytest.mark.skipif(os.environ.get("ASTEROID_LIVE_TESTS") != "1",
                    reason="set ASTEROID_LIVE_TESTS=1 to run network tests")
def test_live_nbody_beats_two_body_vs_horizons():
    """N-body propagation of a fresh catalog orbit must track Horizons far better
    than two-body, a year out (away from any deep flyby)."""
    from asteroid import nbody, observe
    b = fetch.fetch_sbdb("Bennu")
    el = b.to_elements()
    jd = el.epoch + 365.0
    nasa = fetch.fetch_horizons_vector(b.designation or b.name, jd)
    err_two = np.linalg.norm(observe.observe(el, jd).position_helio - nasa) * AU_KM
    r_nb, _ = nbody.propagate_elements(el, jd)
    err_nb = np.linalg.norm(r_nb - nasa) * AU_KM
    assert err_nb < 5_000.0            # km: sub-5000-km a year out
    assert err_nb < err_two / 50.0     # at least 50x better than two-body
