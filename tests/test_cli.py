"""CLI smoke tests (offline; operate on the seeded temp database)."""

import json

import pytest

from asteroid import cli, database


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTEROID_HOME", str(tmp_path))
    monkeypatch.delenv("ASTEROID_DB", raising=False)
    database.reset_db()
    yield


def run(argv):
    return cli.main(argv)


def test_parse_duration():
    assert cli.parse_duration("30d") == pytest.approx(30.0)
    assert cli.parse_duration("12h") == pytest.approx(0.5)
    assert cli.parse_duration("1y") == pytest.approx(365.25)
    with pytest.raises(ValueError):
        cli.parse_duration("nonsense")


def test_parse_range_auto_and_explicit():
    lo, hi = cli.parse_range("2025..2035", 2460000.0)
    assert hi > lo
    lo2, hi2 = cli.parse_range("__AUTO__", 2460000.0)
    assert lo2 < 2460000.0 < hi2


def test_list_runs(capsys):
    assert run(["--list"]) == 0
    out = capsys.readouterr().out
    assert "Apophis" in out and "database" in out.lower()


def test_default_report_offline(capsys):
    assert run(["Apophis", "--date", "2029-04-13", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "Apophis" in out
    assert "Earth distance" in out


def test_info_offline(capsys):
    assert run(["Bennu", "--info", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "ORBITAL ELEMENTS" in out.upper()


def test_json_output(capsys):
    assert run(["Ceres", "--date", "2026-06-29", "--json", "--offline"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"].endswith("Ceres")
    assert payload["state"]["earth_distance_km"] > 0
    assert "ra_deg" in payload["state"]


def test_ephemeris_runs(capsys):
    assert run(["Vesta", "--date", "2026-06-29", "--span", "20d",
                "--step", "10d", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "Ephemeris" in out


def test_approaches_finds_apophis_2029(capsys):
    assert run(["Apophis", "--approaches", "2025..2035", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "close approaches" in out.lower()
    assert "2029-04" in out


def test_interstellar_alias_offline(capsys):
    """The '1I' shorthand resolves to 'Oumuamua from the seed without network."""
    assert run(["1I", "--info", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "Oumuamua" in out


def test_determine_orbit_from_file(tmp_path, capsys):
    """End-to-end: determine an orbit from an observation file, then use it."""
    import numpy as np
    from asteroid import frames
    from asteroid.bodies import earth_position
    from asteroid.propagate import OrbitalElements, state_vector

    truth = OrbitalElements.from_degrees(
        a=0.9223592206975018, e=0.1911492279663492, inc_deg=3.340996879880978,
        node_deg=203.8936514240762, argp_deg=126.6795706895841,
        M0_deg=175.3304026592739, epoch=2461200.5, n_deg=1.112638115271892)
    base = frames.parse_date("2027-03-01")
    lines = []
    for d in (0, 6, 12, 18, 24, 30):
        jd = base + d
        ra, dec = frames.equatorial_to_radec(
            frames.ecliptic_to_equatorial(state_vector(truth, jd)[0] - earth_position(jd)))
        lines.append(f"{frames.format_jd(jd, with_time=False)}, {ra % 360:.6f}, {dec:.6f}")
    obs_file = tmp_path / "newobj.obs"
    obs_file.write_text("\n".join(lines) + "\n")

    assert run(["--observations", str(obs_file), "--as", "TestObj"]) == 0
    out = capsys.readouterr().out
    assert "Orbit determined" in out and "TestObj" in out

    # The determined object is saved and usable by name.
    assert database.get_body("TestObj") is not None
    assert run(["TestObj", "--date", "2027-06-01", "--offline"]) == 0


def test_unknown_offline_errors():
    assert run(["zzz-not-real", "--offline"]) == 1


def test_missing_name_errors():
    assert run([]) == 2
