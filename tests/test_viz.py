"""Smoke tests for the visualization backends."""

import io

import pytest
from rich.console import Console

from asteroid import viz
from asteroid.database import Body


def apophis_body() -> Body:
    return Body(
        key="apophis", name="99942 Apophis", a=0.9223592206975018,
        e=0.1911492279663492, inc_deg=3.340996879880978,
        node_deg=203.8936514240762, argp_deg=126.6795706895841,
        M0_deg=175.3304026592739, epoch=2461200.5, n_deg=1.112638115271892,
        H=19.09, neo=True, pha=True, orbit_class="Aten", fullname="99942 Apophis",
    )


def test_ascii_orbit_renders():
    console = Console(file=io.StringIO(), width=90, force_terminal=False)
    viz.ascii_orbit(apophis_body(), 2462239.0, console)
    out = console.file.getvalue()
    assert "Orbit of" in out
    assert "Sun" in out and "Earth" in out


def test_context_planets_frames_neo_tightly():
    """A near-Earth orbit should not pull in Jupiter (which would zoom out)."""
    planets = viz._context_planets(1.1)
    assert "Jupiter" not in planets
    assert {"Mercury", "Venus", "Earth", "Mars"}.issubset(set(planets))


def test_context_planets_includes_outer_for_distant_orbit():
    planets = viz._context_planets(35.0)   # Halley-like reach
    assert "Jupiter" in planets and "Neptune" in planets


def test_plot_png_creates_file(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "orbit.png"
    viz.plot_png(apophis_body(), 2462239.0, str(out))
    assert out.exists() and out.stat().st_size > 1000


def test_plot_html_creates_file(tmp_path):
    pytest.importorskip("plotly")
    out = tmp_path / "orbit.html"
    viz.plot_html(apophis_body(), 2462239.0, str(out))
    assert out.exists()
    assert "plotly" in out.read_text().lower()
