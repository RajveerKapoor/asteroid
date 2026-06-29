"""Visualizations: an ASCII orbit map, a matplotlib PNG, and a Plotly HTML.

The ASCII map always works (pure stdlib + numpy + rich). The PNG and interactive
HTML need the optional ``[viz]`` extras (matplotlib, plotly); their imports are
lazy and degrade with a clear message if the backend is missing.

All three draw the same scene: the Sun at the origin, the inner planet orbits
for context, the asteroid's full orbit, and the current positions of Earth and
the asteroid in the heliocentric ecliptic frame.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from .bodies import earth_position, planet_elements_at, planet_position, PLANET_ORDER
from .database import Body
from .propagate import orbit_path, position


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "asteroid"


def _examples_dir() -> Path:
    d = Path.cwd() / "examples"
    d.mkdir(exist_ok=True)
    return d


def _context_planets(extent: float) -> list:
    """Planets to draw for context.

    Always the inner four (Mercury-Mars) for a familiar frame, plus any outer
    planet whose orbit is comparable to the asteroid's reach (``extent``, the
    aphelion for bound orbits). This keeps a near-Earth asteroid framed tightly
    while still showing the whole outer system for something like Halley.
    """
    chosen = []
    for name in PLANET_ORDER:
        el = planet_elements_at(name, 2451545.0)
        if name in ("Mercury", "Venus", "Earth", "Mars") or el.perihelion <= extent * 1.15:
            chosen.append(name)
    return chosen or ["Earth"]


def _extent_for(el) -> float:
    """A representative outer radius for choosing context planets / framing."""
    return el.aphelion if el.e < 1.0 else el.perihelion * 4.0


# --------------------------------------------------------------------------- #
# ASCII orbit map
# --------------------------------------------------------------------------- #
def ascii_orbit(body: Body, jd: float, console, width: int = 79) -> None:
    """Render a top-down (ecliptic x-y) ASCII map of the orbit to ``console``."""
    from rich.text import Text

    el = body.to_elements()
    ast_path = orbit_path(el, 720)[:, :2]
    ast_pos = position(el, jd)[:2]
    earth_pos = earth_position(jd)[:2]

    # Earth's orbit as context (sample one year).
    earth_path = np.array([earth_position(jd + d)[:2] for d in range(0, 366, 6)])

    pts = np.vstack([ast_path, earth_path, ast_pos, earth_pos, [0.0, 0.0]])
    max_r = float(np.max(np.abs(pts))) * 1.08
    if max_r <= 0:
        return

    height = width // 2 | 1                  # odd, ~2:1 char aspect
    half_w, half_h = (width - 1) / 2, (height - 1) / 2
    sx = half_w / max_r
    sy = sx * 0.5                            # compress vertically for square cells

    grid = [[" "] * width for _ in range(height)]
    styles = [[""] * width for _ in range(height)]

    def plot(x, y, ch, style, overwrite=True):
        col = int(round(half_w + x * sx))
        row = int(round(half_h - y * sy))
        if 0 <= row < height and 0 <= col < width:
            if overwrite or grid[row][col] == " ":
                grid[row][col] = ch
                styles[row][col] = style

    for x, y in earth_path:
        plot(x, y, "·", "blue", overwrite=False)
    for x, y in ast_path:
        plot(x, y, "•", "bright_cyan", overwrite=False)
    plot(0, 0, "☉", "bold yellow")
    plot(earth_pos[0], earth_pos[1], "⊕", "bold blue")
    plot(ast_pos[0], ast_pos[1], "◉", "bold red")

    console.print()
    title = Text(f"  Orbit of {body.name}  (top-down ecliptic view, "
                 f"{max_r:.2f} AU across)", style="bold")
    console.print(title)
    for row in range(height):
        line = Text()
        for col in range(width):
            line.append(grid[row][col], style=styles[row][col] or None)
        console.print(line)
    legend = Text("  ")
    legend.append("☉ Sun", style="yellow")
    legend.append("   ⊕ Earth", style="blue")
    legend.append("   ◉ " + body.name, style="red")
    legend.append("   • orbit", style="bright_cyan")
    console.print(legend)


# --------------------------------------------------------------------------- #
# Animated ASCII trajectory
# --------------------------------------------------------------------------- #
def animate_orbit(body: Body, jd: float, console, n_frames: int = 84,
                  fps: int = 22, width: int = 79) -> None:
    """Animate the asteroid tracing its orbit in the terminal.

    A moving marker sweeps along the path from ``jd`` forward (one revolution for
    bound orbits), leaving a bright trail so the trajectory draws itself in, with
    Earth moving on its own orbit for scale.
    """
    import time
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    el = body.to_elements()
    guide = orbit_path(el, 720)[:, :2]
    earth_ring = np.array([earth_position(jd + d)[:2] for d in range(0, 366, 9)])
    span = el.period_days if (el.e < 1.0 and math.isfinite(el.period_days)) else 1460.0
    times = [jd + span * k / n_frames for k in range(n_frames + 1)]
    ast_xy = [position(el, t)[:2] for t in times]
    earth_xy = [earth_position(t)[:2] for t in times]

    pts = np.vstack([guide, earth_ring, [0.0, 0.0]])
    max_r = float(np.max(np.abs(pts))) * 1.08
    if max_r <= 0:
        return
    width = max(40, min(width, console.width - 4))
    height = width // 2 | 1
    half_w, half_h = (width - 1) / 2, (height - 1) / 2
    sx, sy = half_w / max_r, half_w / max_r * 0.5

    def build(i: int):
        grid = [[" "] * width for _ in range(height)]
        sty = [[""] * width for _ in range(height)]

        def put(x, y, ch, style, over=True):
            c = int(round(half_w + x * sx))
            r = int(round(half_h - y * sy))
            if 0 <= r < height and 0 <= c < width and (over or grid[r][c] == " "):
                grid[r][c], sty[r][c] = ch, style

        for x, y in earth_ring:
            put(x, y, "·", "blue", over=False)
        for x, y in guide:
            put(x, y, "·", "bright_black", over=False)
        for k in range(i + 1):                     # the trail drawn so far
            put(ast_xy[k][0], ast_xy[k][1], "•", "bright_cyan")
        put(0, 0, "☉", "bold yellow")
        put(earth_xy[i][0], earth_xy[i][1], "⊕", "bold bright_blue")
        put(ast_xy[i][0], ast_xy[i][1], "◉", "bold bright_red")

        text = Text()
        for r in range(height):
            for c in range(width):
                text.append(grid[r][c], style=sty[r][c] or None)
            if r < height - 1:
                text.append("\n")
        sub = Text.assemble(
            (f"  t+{times[i] - jd:6.1f} d   ", "dim"),
            ("☉ Sun  ", "yellow"), ("⊕ Earth  ", "bright_blue"),
            ("◉ " + body.name, "bright_red"))
        return Panel(text, title=Text(f"☄ {body.name} — orbit trace "
                     f"({max_r:.2f} AU across)", style="bold bright_cyan"),
                     subtitle=sub, border_style="bright_cyan", padding=(0, 1))

    console.print()
    with Live(build(0), console=console, refresh_per_second=fps, screen=False) as live:
        for i in range(1, len(times)):
            time.sleep(1.0 / fps)
            live.update(build(i))


# --------------------------------------------------------------------------- #
# matplotlib PNG
# --------------------------------------------------------------------------- #
def plot_png(body: Body, jd: float, filename: str) -> str:
    """Render a 2D + 3D matplotlib figure of the orbit; returns the path."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for --plot; install with: pip install '.[viz]'"
        ) from exc

    el = body.to_elements()
    path = orbit_path(el, 720)
    ast_pos = position(el, jd)
    planets = _context_planets(_extent_for(el))

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"{body.fullname or body.name} — heliocentric orbit", fontsize=14)

    # --- top-down (x-y) ---
    ax = fig.add_subplot(1, 2, 1)
    ax.set_facecolor("#05060a")
    for name in planets:
        pp = orbit_path(planet_elements_at(name, jd), 360)
        ax.plot(pp[:, 0], pp[:, 1], lw=0.6, color="#5a6172")
        pos = planet_position(name, jd)
        ax.scatter(pos[0], pos[1], s=14, color="#9aa3b2", zorder=3)
    ax.plot(path[:, 0], path[:, 1], lw=1.3, color="#37d0ff", label="orbit")
    ax.scatter([0], [0], s=120, color="#ffd23f", marker="*", zorder=5, label="Sun")
    earth = earth_position(jd)
    ax.scatter(earth[0], earth[1], s=30, color="#3aa0ff", zorder=5, label="Earth")
    ax.scatter(ast_pos[0], ast_pos[1], s=45, color="#ff4d4d", zorder=6, label=body.name)
    ax.set_aspect("equal")
    ax.set_xlabel("x (AU)"); ax.set_ylabel("y (AU)")
    ax.set_title("top-down (ecliptic plane)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.3)
    ax.grid(alpha=0.15)

    # --- 3D ---
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    ax3.plot(path[:, 0], path[:, 1], path[:, 2], lw=1.2, color="#37d0ff")
    for name in planets:
        pp = orbit_path(planet_elements_at(name, jd), 360)
        ax3.plot(pp[:, 0], pp[:, 1], pp[:, 2], lw=0.5, color="#5a6172")
    ax3.scatter([0], [0], [0], s=120, color="#ffd23f", marker="*")
    ax3.scatter(earth[0], earth[1], earth[2], s=30, color="#3aa0ff")
    ax3.scatter(ast_pos[0], ast_pos[1], ast_pos[2], s=45, color="#ff4d4d")
    ax3.set_xlabel("x (AU)"); ax3.set_ylabel("y (AU)"); ax3.set_zlabel("z (AU)")
    ax3.set_title(f"3D (inclination {body.inc_deg:.1f}°)")

    fig.tight_layout()
    fig.savefig(filename, dpi=130, facecolor="white")
    plt.close(fig)
    return filename


# --------------------------------------------------------------------------- #
# Plotly interactive HTML
# --------------------------------------------------------------------------- #
def plot_html(body: Body, jd: float, filename: str) -> str:
    """Render an interactive, rotatable 3D Plotly scene; returns the path."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "plotly is required for --html; install with: pip install '.[viz]'"
        ) from exc

    el = body.to_elements()
    path = orbit_path(el, 720)
    ast_pos = position(el, jd)
    earth = earth_position(jd)
    planets = _context_planets(_extent_for(el))

    traces = []
    for name in planets:
        pp = orbit_path(planet_elements_at(name, jd), 360)
        traces.append(go.Scatter3d(
            x=pp[:, 0], y=pp[:, 1], z=pp[:, 2], mode="lines",
            line=dict(color="#5a6172", width=2), name=name, hoverinfo="name"))
    traces.append(go.Scatter3d(
        x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines",
        line=dict(color="#37d0ff", width=4), name=f"{body.name} orbit"))
    traces.append(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                               marker=dict(size=7, color="#ffd23f"), name="Sun"))
    traces.append(go.Scatter3d(x=[earth[0]], y=[earth[1]], z=[earth[2]], mode="markers",
                               marker=dict(size=5, color="#3aa0ff"), name="Earth"))
    traces.append(go.Scatter3d(
        x=[ast_pos[0]], y=[ast_pos[1]], z=[ast_pos[2]], mode="markers",
        marker=dict(size=6, color="#ff4d4d"), name=body.name))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{body.fullname or body.name} — heliocentric orbit (J2000 ecliptic)",
        template="plotly_dark",
        scene=dict(xaxis_title="x (AU)", yaxis_title="y (AU)", zaxis_title="z (AU)",
                   aspectmode="data"),
        showlegend=True,
    )
    fig.write_html(filename, include_plotlyjs="cdn")
    return filename


# --------------------------------------------------------------------------- #
# CLI hook
# --------------------------------------------------------------------------- #
def handle_visualizations(body: Body, jd: float, args, console) -> None:
    """Dispatch --ascii / --plot / --html based on parsed CLI args."""
    if args.ascii:
        ascii_orbit(body, jd, console)

    if args.plot is not None:
        target = args.plot if args.plot != "__AUTO__" else \
            str(_examples_dir() / f"{_safe_name(body.name)}_orbit.png")
        try:
            out = plot_png(body, jd, target)
            console.print(f"[green]Saved plot:[/green] {out}")
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")

    if args.html is not None:
        target = args.html if args.html != "__AUTO__" else \
            str(_examples_dir() / f"{_safe_name(body.name)}_orbit.html")
        try:
            out = plot_html(body, jd, target)
            console.print(f"[green]Saved interactive view:[/green] {out}")
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
