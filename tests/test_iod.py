"""Tests for orbit determination (Gauss's method + differential correction).

The core check is self-consistency: generate synthetic observations from a known
orbit with the forward model, then confirm the inverse pipeline recovers that
orbit. With clean data recovery is exact; with added noise it should degrade
gracefully, like real astrometry.
"""

import numpy as np
import pytest

from asteroid import frames, iod
from asteroid.bodies import earth_position
from asteroid.constants import RAD2DEG, SPEED_OF_LIGHT_AU_D, AU_KM
from asteroid.propagate import OrbitalElements, state_vector


def truth_apophis() -> OrbitalElements:
    return OrbitalElements.from_degrees(
        a=0.9223592206975018, e=0.1911492279663492, inc_deg=3.340996879880978,
        node_deg=203.8936514240762, argp_deg=126.6795706895841,
        M0_deg=175.3304026592739, epoch=2461200.5, n_deg=1.112638115271892)


def truth_mainbelt() -> OrbitalElements:
    return OrbitalElements.from_degrees(
        a=2.7656, e=0.0797, inc_deg=10.59, node_deg=80.27,
        argp_deg=73.6, M0_deg=130.0, epoch=2461000.5)


def synth_obs(truth, jds, noise_arcsec=0.0, light_time=False, seed=0):
    rng = np.random.default_rng(seed)
    obs = []
    for jd in jds:
        r = state_vector(truth, jd)[0]
        earth = earth_position(jd)
        if light_time:
            for _ in range(2):
                d = np.linalg.norm(r - earth)
                r = state_vector(truth, jd - d / SPEED_OF_LIGHT_AU_D)[0]
        ra, dec = frames.equatorial_to_radec(frames.ecliptic_to_equatorial(r - earth))
        if noise_arcsec:
            n = noise_arcsec / 3600.0
            ra += rng.normal(0, n) / max(np.cos(np.radians(dec)), 0.1)
            dec += rng.normal(0, n)
        obs.append(iod.Observation(jd=jd, ra_deg=ra % 360.0, dec_deg=dec))
    return obs


def test_radec_ecliptic_unit_is_unit_and_consistent():
    u = iod.radec_to_ecliptic_unit(123.4, -42.0)
    assert np.linalg.norm(u) == pytest.approx(1.0, abs=1e-12)
    back = frames.equatorial_to_radec(frames.ecliptic_to_equatorial(u))
    assert back[0] == pytest.approx(123.4, abs=1e-9)
    assert back[1] == pytest.approx(-42.0, abs=1e-9)


def test_recover_apophis_noiseless():
    truth = truth_apophis()
    base = frames.parse_date("2027-03-01")
    obs = synth_obs(truth, [base + d for d in (0, 6, 12, 18, 24, 30)])
    sol = iod.determine_orbit(obs, light_time=False)
    assert sol.rms_arcsec < 1e-2
    assert sol.elements.a == pytest.approx(truth.a, abs=1e-4)
    assert sol.elements.e == pytest.approx(truth.e, abs=1e-4)
    assert sol.elements.inc * RAD2DEG == pytest.approx(truth.inc * RAD2DEG, abs=1e-3)


def test_recover_main_belt_noiseless():
    truth = truth_mainbelt()
    base = frames.parse_date("2026-01-01")
    obs = synth_obs(truth, [base + d for d in (0, 20, 40, 60, 80)])
    sol = iod.determine_orbit(obs, light_time=False)
    assert sol.rms_arcsec < 1e-1
    assert sol.elements.a == pytest.approx(truth.a, rel=1e-3)
    assert sol.elements.e == pytest.approx(truth.e, abs=2e-3)


def test_predicted_position_matches_truth_noiseless():
    truth = truth_apophis()
    base = frames.parse_date("2027-03-01")
    obs = synth_obs(truth, [base + d for d in (0, 8, 16, 24, 32)])
    sol = iod.determine_orbit(obs, light_time=False)
    jd_future = base + 200
    err_km = np.linalg.norm(
        state_vector(truth, jd_future)[0]
        - state_vector(sol.elements, jd_future)[0]) * AU_KM
    assert err_km < 1000.0  # essentially exact


def test_noise_degrades_gracefully():
    truth = truth_apophis()
    base = frames.parse_date("2027-03-01")
    jds = [base + d for d in (0, 5, 10, 15, 20, 25, 30, 35, 40)]
    obs = synth_obs(truth, jds, noise_arcsec=0.5, light_time=True, seed=42)
    sol = iod.determine_orbit(obs, light_time=True)
    # Fit residual should be on the order of the input noise, orbit roughly right.
    assert sol.rms_arcsec < 3.0
    assert sol.elements.a == pytest.approx(truth.a, abs=0.02)
    assert sol.elements.e == pytest.approx(truth.e, abs=0.02)


def test_too_few_observations_raises():
    truth = truth_apophis()
    base = frames.parse_date("2027-03-01")
    obs = synth_obs(truth, [base, base + 10])
    with pytest.raises(ValueError):
        iod.determine_orbit(obs)


def test_parse_angle_sexagesimal_and_decimal():
    # RA sexagesimal hours -> degrees.
    assert iod._parse_angle("12:00:00", is_ra=True) == pytest.approx(180.0)
    assert iod._parse_angle("06:00:00", is_ra=True) == pytest.approx(90.0)
    # Dec sexagesimal degrees, signed.
    assert iod._parse_angle("-23:30:00", is_ra=False) == pytest.approx(-23.5)
    # Decimal passthrough.
    assert iod._parse_angle("145.25", is_ra=False) == pytest.approx(145.25)


def test_parse_observation_file_roundtrip(tmp_path):
    f = tmp_path / "obs.txt"
    f.write_text(
        "# a comment\n"
        "2027-03-02, 02:40:26.7, +12:11:27.4\n"
        "2027-03-05, 43.2362, 13.1104\n"      # decimal degrees form
        "\n"
    )
    obs = iod.parse_observation_file(str(f))
    assert len(obs) == 2
    assert obs[0].ra_deg == pytest.approx(15 * (2 + 40/60 + 26.7/3600), abs=1e-6)
    assert obs[1].dec_deg == pytest.approx(13.1104, abs=1e-9)
