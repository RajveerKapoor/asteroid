"""Command-line interface: ``asteroid <name> --date <date> [options]``.

This is the orchestration layer. It resolves a body (local database first, then
a live JPL fetch), propagates it to the requested time, and renders a rich
terminal report. Flags add an ephemeris table, a close-approach scan, a NASA
Horizons validation, JSON output, or visualizations.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from typing import Optional

# LibreSSL on macOS triggers a noisy urllib3 warning on import; it is harmless.
warnings.filterwarnings("ignore", message=r".*OpenSSL.*")

import numpy as np
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Cohesive palette so every view shares one look.
ACCENT = "bright_cyan"
SUN = "yellow"
EARTH = "bright_blue"
BODY = "bright_red"

from . import __version__, frames, observe
from .constants import AU_KM
from .database import Body, get_body, list_bodies, upsert_body
from .propagate import OrbitalElements, ephemeris

console = Console()
err_console = Console(stderr=True)

_DURATION_UNITS = {"h": 1.0 / 24.0, "d": 1.0, "w": 7.0, "m": 30.4375, "y": 365.25}


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="asteroid",
        description="Load an asteroid from a database and compute its trajectory.",
        epilog="examples:\n"
               "  asteroid Apophis --date 2029-04-13\n"
               "  asteroid Ceres --span 90d --step 5d\n"
               "  asteroid Apophis --approaches 2025..2035\n"
               "  asteroid Bennu --validate\n"
               "  asteroid --list",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("name", nargs="?", help="asteroid name or designation")
    p.add_argument("--name", dest="name_opt", help="asteroid name (alternative form)")
    p.add_argument("--date", default=None,
                   help="date: YYYY-MM-DD[ HH:MM], 'now', or JD###### (default: now)")
    p.add_argument("--span", help="ephemeris span, e.g. 30d, 6m, 1y")
    p.add_argument("--step", default="1d", help="ephemeris step, e.g. 1d, 12h (default 1d)")
    p.add_argument("--approaches", nargs="?", const="__AUTO__", metavar="A..B",
                   help="scan Earth close approaches over a range (default date +-5y)")
    p.add_argument("--validate", action="store_true",
                   help="compare our position against NASA Horizons")
    p.add_argument("--ascii", action="store_true", help="draw an ASCII orbit map")
    p.add_argument("--plot", nargs="?", const="__AUTO__", metavar="FILE",
                   help="render a matplotlib PNG of the orbit")
    p.add_argument("--html", nargs="?", const="__AUTO__", metavar="FILE",
                   help="render an interactive 3D Plotly HTML of the orbit")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p.add_argument("--list", action="store_true", help="list bodies in the local database")
    p.add_argument("--info", action="store_true", help="show a full parameter sheet")
    p.add_argument("--update", nargs="*", metavar="NAME",
                   help="refresh/add bodies from JPL (no args = refresh all)")
    p.add_argument("--determine", action="store_true",
                   help="determine the orbit from scratch using observations fetched "
                        "online from the Minor Planet Center")
    p.add_argument("--observations", "--obs", dest="observations", nargs="?",
                   const="__ONLINE__", metavar="FILE",
                   help="determine an orbit from a local (date, RA, Dec) file; "
                        "with no file, fetch observations online (same as --determine)")
    p.add_argument("--as", dest="as_name", metavar="NAME",
                   help="name to give an orbit determined from observations")
    p.add_argument("--animate", action="store_true",
                   help="animate the asteroid tracing its orbit in the terminal")
    p.add_argument("--offline", action="store_true", help="never use the network")
    p.add_argument("--version", action="version", version=f"asteroid-trajectory {__version__}")
    return p


def parse_duration(text: str) -> float:
    """Parse a duration like ``30d``, ``12h``, ``6m``, ``1y`` into days."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([hdwmy])\s*", text.lower())
    if not m:
        raise ValueError(f"bad duration {text!r}; use forms like 30d, 12h, 6m, 1y")
    return float(m.group(1)) * _DURATION_UNITS[m.group(2)]


def parse_date_or_year(text: str) -> float:
    """Parse a date, or a bare 4-digit year as that year's Jan 1."""
    text = text.strip()
    if re.fullmatch(r"\d{4}", text):
        return frames.parse_date(f"{text}-01-01")
    return frames.parse_date(text)


def parse_range(text: str, center_jd: float) -> tuple[float, float]:
    """Parse ``A..B`` into ``(jd_start, jd_end)``; ``__AUTO__`` => center +-5y."""
    if text == "__AUTO__":
        return center_jd - 5 * 365.25, center_jd + 5 * 365.25
    if ".." not in text:
        raise ValueError(f"bad range {text!r}; use A..B (e.g. 2025..2035)")
    lo, hi = text.split("..", 1)
    return parse_date_or_year(lo), parse_date_or_year(hi)


# --------------------------------------------------------------------------- #
# Body resolution
# --------------------------------------------------------------------------- #
def resolve_body(name: str, offline: bool) -> Body:
    """Find a body in the local DB, falling back to a live JPL fetch."""
    body = get_body(name)
    if body is not None:
        return body
    if offline:
        raise LookupError(f"{name!r} not in local database (and --offline is set)")
    # Live fetch.
    from .fetch import fetch_sbdb, FetchError
    try:
        body = fetch_sbdb(name)
    except FetchError as exc:
        raise LookupError(str(exc)) from exc
    upsert_body(body)
    return body


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def _badges(body: Body) -> Text:
    """Coloured classification chips (orbit class · NEO · PHA · hyperbolic)."""
    specs = []
    if body.orbit_class:
        specs.append((body.orbit_class, "cyan"))
    if body.neo:
        specs.append(("NEO", "yellow"))
    if body.pha:
        specs.append(("PHA", "red"))
    if body.e >= 1.0:
        specs.append(("HYPERBOLIC", "magenta"))
    t = Text()
    for i, (txt, color) in enumerate(specs):
        if i:
            t.append(" ")
        t.append(f" {txt} ", style=f"bold white on {color}")
    return t


def _title(body: Body) -> Text:
    return Text.assemble(("☄  ", SUN), (body.fullname or body.name, f"bold {ACCENT}"))


def _section(label: str, note: str = "") -> Text:
    t = Text.assemble((f"  {label}", f"bold {ACCENT}"))
    if note:
        t.append(f"   {note}", style="dim")
    return t


def _elements_grid(body: Body) -> Table:
    el = body.to_elements()
    q = body.q if body.q is not None else el.perihelion
    period = "—" if math.isinf(el.period_days) else f"{el.period_days / 365.25:.3f} yr"
    aph = "∞ (unbound)" if body.e >= 1.0 else f"{el.aphelion:.4f} AU"
    g = Table.grid(padding=(0, 1))
    for _ in range(3):
        g.add_column(justify="right", style="dim", min_width=6)
        g.add_column(justify="left", style="white", min_width=14)
    g.add_row("a", f"{body.a:.6f} AU", "e", f"{body.e:.6f}", "i", f"{body.inc_deg:.3f}°")
    g.add_row("Ω", f"{body.node_deg:.3f}°", "ω", f"{body.argp_deg:.3f}°",
              "M", f"{body.M0_deg:.3f}°")
    g.add_row("q", f"{q:.4f} AU", "Q", aph, "P", period)
    return g


def _state_grid(body: Body, jd: float) -> Table:
    el = body.to_elements()
    o = observe.observe(el, jd, H=body.H)
    g = Table.grid(padding=(0, 1))
    g.add_column(justify="right", style="dim", min_width=14)
    g.add_column(justify="left")
    g.add_row("Sun distance", f"[white]{o.r_helio:.5f} AU[/white]  "
                              f"[dim]({o.r_helio * AU_KM:,.0f} km)[/dim]")
    ld = o.lunar_distances
    near = f"  ·  [bold {BODY}]{ld:.2f} lunar distances[/]" if ld < 60 else ""
    g.add_row("Earth distance", f"[white]{o.delta:.5f} AU[/white]  "
                                f"[dim]({o.delta_km:,.0f} km)[/dim]{near}")
    g.add_row("Sky RA / Dec", f"[white]{frames.format_ra(o.ra)}[/white]   "
                              f"[white]{frames.format_dec(o.dec)}[/white]")
    extra = (f"elong [white]{o.elongation:.1f}°[/]   phase [white]{o.phase_angle:.1f}°[/]"
             f"   speed [white]{o.speed_helio_km_s:.2f} km/s[/]")
    if o.magnitude is not None:
        vis = ("naked eye" if o.magnitude < 6 else "binoculars" if o.magnitude < 10
               else "small scope" if o.magnitude < 15 else "large scope")
        extra += f"   mag [white]{o.magnitude:.1f}[/] [dim]({vis})[/dim]"
    g.add_row("Geometry", extra)
    return g


def next_close_approach(body: Body, jd: float, horizon_days: float = 4 * 365.25):
    """The chronologically next Earth close approach at/after ``jd`` (or None)."""
    el = body.to_elements()
    coarse = 1.0 if body.neo else 3.0
    cas = observe.close_approaches(el, jd, jd + horizon_days, coarse_step_days=coarse)
    upcoming = sorted((c for c in cas if c.jd >= jd - 1), key=lambda c: c.jd)
    return upcoming[0] if upcoming else None


def _physical_text(body: Body) -> Text:
    bits = []
    if body.H is not None:
        bits.append(f"H {body.H:.2f}")
    if body.diameter_km is not None:
        bits.append(f"⌀ {body.diameter_km:g} km")
    if body.albedo is not None:
        bits.append(f"albedo {body.albedo:.3f}")
    if body.moid is not None:
        bits.append(f"MOID {body.moid:.5f} AU")
    return Text.assemble(("       Physical", "dim"), ("   " + "  ·  ".join(bits), "white")) \
        if bits else Text("")


def render_report(body: Body, jd: float, show_approach: bool = True) -> Panel:
    """One cohesive panel: identity, elements, live state, next approach, physical."""
    epoch = f"epoch {frames.format_jd(body.epoch, with_time=False)} · JD {body.epoch:.1f}"
    blocks = []
    badges = _badges(body)
    if str(badges):
        blocks += [badges, Text("")]
    blocks += [
        _section("ORBITAL ELEMENTS", epoch),
        _elements_grid(body),
        Text(""),
        _section("POSITION & SKY", frames.format_jd(jd)),
        _state_grid(body, jd),
    ]
    if show_approach:
        ca = next_close_approach(body, jd)
        if ca is not None:
            ld = ca.lunar_distances
            tone = BODY if ld < 20 else "white"
            blocks += [Text(""), Text.assemble(
                ("  NEXT APPROACH", f"bold {ACCENT}"),
                (f"   {frames.format_jd(ca.jd, with_time=False)}  ·  ", "dim"),
                (f"{ca.distance_km:,.0f} km ({ld:.2f} LD)", f"bold {tone}"))]
    phys = _physical_text(body)
    if str(phys):
        blocks += [Text(""), phys]

    return Panel(Group(*blocks), title=_title(body), title_align="left",
                 subtitle=Text(f"source: {body.source or '?'}", style="dim"),
                 subtitle_align="right", box=box.ROUNDED, border_style=ACCENT,
                 padding=(1, 2))


def render_elements_table(body: Body) -> Panel:
    """Standalone elements panel (used by --info and orbit determination)."""
    epoch = f"epoch {frames.format_jd(body.epoch, with_time=False)} · JD {body.epoch:.1f}"
    blocks = []
    badges = _badges(body)
    if str(badges):
        blocks += [badges, Text("")]
    blocks += [_section("ORBITAL ELEMENTS", epoch), _elements_grid(body)]
    return Panel(Group(*blocks), title=_title(body), title_align="left",
                 box=box.ROUNDED, border_style=ACCENT, padding=(1, 2))


# --------------------------------------------------------------------------- #
# Command handlers
# --------------------------------------------------------------------------- #
def cmd_list() -> int:
    bodies = list_bodies()
    table = Table(title=f"Local asteroid database  ({len(bodies)} bodies)",
                  title_style="bold cyan")
    table.add_column("Name", style="cyan")
    table.add_column("a (AU)", justify="right")
    table.add_column("e", justify="right")
    table.add_column("i (°)", justify="right")
    table.add_column("period", justify="right")
    table.add_column("class", style="yellow")
    table.add_column("flags", style="magenta")
    for b in bodies:
        el = b.to_elements()
        period = "—" if el.period_days == float("inf") else f"{el.period_days/365.25:.2f} yr"
        flags = " ".join(f for f, on in (("NEO", b.neo), ("PHA", b.pha),
                                         ("hyp", b.e >= 1.0)) if on)
        table.add_row(b.name, f"{b.a:.3f}", f"{b.e:.3f}", f"{b.inc_deg:.2f}",
                      period, b.orbit_class, flags)
    console.print(table)
    return 0


def cmd_update(names) -> int:
    from .fetch import fetch_sbdb, FetchError
    if not names:
        names = [b.name for b in list_bodies()]
        console.print(f"[dim]Refreshing {len(names)} bodies from JPL SBDB...[/dim]")
    ok = 0
    for n in names:
        try:
            body = fetch_sbdb(n)
        except FetchError as exc:
            err_console.print(f"[red]  fail[/red] {n}: {exc}")
            continue
        upsert_body(body)
        console.print(f"[green]  ok[/green]   {body.name}  (epoch JD {body.epoch:.1f})")
        ok += 1
    console.print(f"[bold]Updated {ok}/{len(names)} bodies.[/bold]")
    return 0 if ok else 1


def cmd_determine(args) -> Optional[Body]:
    """Determine an orbit from observations (online from the MPC, or a local file).

    The inverse workflow: instead of loading a known orbit, compute one from raw
    sky observations (Gauss's method + least-squares refinement), then save it so
    every other feature (report, ephemeris, approaches, plots) works on it.
    """
    from pathlib import Path
    from . import iod

    name_query = args.name_opt or args.name
    online = args.determine or args.observations == "__ONLINE__"

    if online:
        if not name_query:
            err_console.print("[red]error:[/red] --determine needs an object name, "
                              "e.g. `asteroid Apophis --determine`.")
            return None
        name = args.as_name or name_query
        from .fetch import fetch_mpc_observations, FetchError
        try:
            with console.status(f"[{ACCENT}]Fetching observations for "
                                f"{name_query!r} from the Minor Planet Center…",
                                spinner="earth"):
                observations = fetch_mpc_observations(name_query)
        except FetchError as exc:
            err_console.print(f"[red]could not fetch observations:[/red] {exc}")
            return None
        source_note = f"MPC, {len(observations)} obs"
    else:
        try:
            observations = iod.parse_observation_file(args.observations)
        except (OSError, ValueError) as exc:
            err_console.print(f"[red]error reading observations:[/red] {exc}")
            return None
        name = args.as_name or Path(args.observations).stem
        source_note = f"file {Path(args.observations).name}"

    try:
        with console.status(f"[{ACCENT}]Determining orbit "
                            f"(Gauss's method → least-squares)…", spinner="dots"):
            sol = iod.determine_orbit(observations, name=name)
    except ValueError as exc:
        err_console.print(f"[red]orbit determination failed:[/red] {exc}")
        return None

    body = Body.from_elements(sol.elements, name=name,
                              source=f"computed from {source_note}")
    upsert_body(body)

    quality, qstyle = (("excellent", "green") if sol.rms_arcsec < 1
                       else ("good", "yellow") if sol.rms_arcsec < 3
                       else ("weak — short arc / noisy data", "red"))
    summary = Group(
        Text.assemble(("  Observations  ", "dim"),
                      (f"{sol.n_obs} over a {sol.arc_days:.0f}-day arc", "white")),
        Text.assemble(("  Fit residual  ", "dim"),
                      (f"{sol.rms_arcsec:.3f}\" RMS", f"bold {qstyle}"),
                      (f"  ({sol.iterations} iterations · {quality})", "dim")),
        Text.assemble(("  Epoch         ", "dim"),
                      (f"{frames.format_jd(sol.epoch, with_time=False)} "
                       f"(JD {sol.epoch:.4f})", "white")),
    )
    console.print()
    console.print(Panel(summary, title=Text(f"✓ Orbit determined: {name}",
                        style="bold green"), title_align="left",
                        box=box.ROUNDED, border_style="green", padding=(1, 2)))
    console.print(render_elements_table(body))
    console.print(f"[dim]Saved — you can now run [/dim][{ACCENT}]asteroid \"{name}\" "
                  f"--approaches[/]  [dim]·[/dim]  [{ACCENT}]--plot[/]  "
                  f"[dim]· … on it.[/dim]")
    return body


def cmd_info(body: Body) -> int:
    console.print()
    console.print(render_elements_table(body))
    phys = _physical_text(body)
    if str(phys):
        console.print(phys)
    return 0


def cmd_report(body: Body, jd: float) -> int:
    console.print()
    console.print(render_report(body, jd))
    return 0


def cmd_ephemeris(body: Body, jd_start: float, span_days: float, step_days: float) -> int:
    el = body.to_elements()
    samples = ephemeris(el, jd_start, jd_start + span_days, step_days)
    table = Table(title=f"Ephemeris for {body.name}  "
                        f"({frames.format_jd(jd_start, with_time=False)} "
                        f"+ {span_days:g} d, step {step_days:g} d)",
                  title_style="bold cyan")
    table.add_column("Date (UTC)", style="cyan")
    table.add_column("r (AU)", justify="right")
    table.add_column("Δ (AU)", justify="right")
    table.add_column("RA", justify="right")
    table.add_column("Dec", justify="right")
    table.add_column("mag", justify="right")
    table.add_column("v (km/s)", justify="right")
    for s in samples:
        obs = observe.observe(el, s.jd, H=body.H)
        mag = f"{obs.magnitude:.1f}" if obs.magnitude is not None else "—"
        table.add_row(frames.format_jd(s.jd, with_time=False),
                      f"{obs.r_helio:.4f}", f"{obs.delta:.4f}",
                      frames.format_ra(obs.ra), frames.format_dec(obs.dec),
                      mag, f"{obs.speed_helio_km_s:.2f}")
    console.print(table)
    return 0


def cmd_approaches(body: Body, rng: str, center_jd: float) -> int:
    el = body.to_elements()
    jd_start, jd_end = parse_range(rng, center_jd)
    span_years = (jd_end - jd_start) / 365.25
    # Fine sampling for NEOs (fast flybys), coarser for slow distant bodies.
    coarse = 0.5 if body.neo else 2.0
    console.print(f"[dim]Scanning {span_years:.1f} yr "
                  f"({frames.format_jd(jd_start, with_time=False)} → "
                  f"{frames.format_jd(jd_end, with_time=False)}) "
                  f"at {coarse:g}-day resolution...[/dim]")
    approaches = observe.close_approaches(el, jd_start, jd_end, coarse_step_days=coarse)
    if not approaches:
        console.print("[yellow]No close-approach minima found in range.[/yellow]")
        return 0
    table = Table(title=f"Earth close approaches · {body.name}", title_style="bold cyan")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Closest approach (UTC)", style="cyan")
    table.add_column("Distance (AU)", justify="right")
    table.add_column("Distance (km)", justify="right")
    table.add_column("Lunar dist", justify="right")
    for i, ca in enumerate(approaches[:15], 1):
        ld = ca.lunar_distances
        style = "bold red" if ld < 1 else "yellow" if ld < 20 else None
        table.add_row(str(i), frames.format_jd(ca.jd), f"{ca.distance_au:.6f}",
                      f"{ca.distance_km:,.0f}", f"{ld:.2f} LD",
                      style=style)
    console.print(table)
    console.print("[dim]Note: two-body model — for deep flybys compare with "
                  "`--validate` against NASA Horizons.[/dim]")
    return 0


def cmd_validate(body: Body, jd: float) -> int:
    from .fetch import fetch_horizons_vector, FetchError
    el = body.to_elements()
    ours = observe.observe(el, jd).position_helio
    # Horizons small-body lookup wants the IAU number / packed designation
    # (e.g. "99942"), not the 8-digit SPK-ID.
    query = body.designation or body.name
    console.print(f"[dim]Querying NASA Horizons for {body.name} at "
                  f"{frames.format_jd(jd)}...[/dim]")
    try:
        nasa = fetch_horizons_vector(query, jd)
    except FetchError as exc:
        err_console.print(f"[red]Horizons validation failed:[/red] {exc}")
        return 1
    diff = ours - nasa
    dist_km = float(np.linalg.norm(diff)) * AU_KM
    r_km = float(np.linalg.norm(nasa)) * AU_KM
    rel = dist_km / r_km if r_km else 0.0
    table = Table(title="Validation vs NASA Horizons (heliocentric ecliptic)",
                  title_style="bold cyan", show_header=True)
    table.add_column("", style="dim")
    table.add_column("X (AU)", justify="right")
    table.add_column("Y (AU)", justify="right")
    table.add_column("Z (AU)", justify="right")
    table.add_row("our two-body", f"{ours[0]:.8f}", f"{ours[1]:.8f}", f"{ours[2]:.8f}")
    table.add_row("NASA Horizons", f"{nasa[0]:.8f}", f"{nasa[1]:.8f}", f"{nasa[2]:.8f}")
    console.print(table)
    console.print(f"[bold]Position difference:[/bold] {dist_km:,.0f} km "
                  f"({rel*100:.4f}% of heliocentric distance)")
    return 0


def observation_to_dict(body: Body, jd: float) -> dict:
    el = body.to_elements()
    obs = observe.observe(el, jd, H=body.H)
    return {
        "name": body.name,
        "fullname": body.fullname,
        "jd": jd,
        "utc": frames.format_jd(jd),
        "elements": {
            "a_au": body.a, "e": body.e, "i_deg": body.inc_deg,
            "node_deg": body.node_deg, "argp_deg": body.argp_deg,
            "M0_deg": body.M0_deg, "epoch_jd": body.epoch,
            "period_days": None if el.period_days == float("inf") else el.period_days,
        },
        "state": {
            "r_helio_au": obs.r_helio,
            "earth_distance_au": obs.delta,
            "earth_distance_km": obs.delta_km,
            "helio_ecl_lon_deg": obs.helio_lon,
            "helio_ecl_lat_deg": obs.helio_lat,
            "ra_deg": obs.ra, "dec_deg": obs.dec,
            "elongation_deg": obs.elongation,
            "phase_angle_deg": obs.phase_angle,
            "speed_km_s": obs.speed_helio_km_s,
            "apparent_magnitude": obs.magnitude,
            "position_helio_au": obs.position_helio.tolist(),
        },
        "flags": {"neo": body.neo, "pha": body.pha,
                  "orbit_class": body.orbit_class},
        "source": body.source,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Standalone commands that need no target body.
    if args.list:
        return cmd_list()
    if args.update is not None:
        return cmd_update(args.update)

    # Orbit determination (online MPC or local file) yields a body the rest can use.
    if args.observations or args.determine:
        body = cmd_determine(args)
        if body is None:
            return 1
    else:
        name = args.name_opt or args.name
        if not name:
            err_console.print("[red]error:[/red] provide an asteroid name "
                              "(e.g. `asteroid Apophis`), a file via "
                              "--observations, or use --list.")
            return 2
        try:
            body = resolve_body(name, args.offline)
        except LookupError as exc:
            err_console.print(f"[red]error:[/red] {exc}")
            return 1

    try:
        jd = frames.now_jd() if args.date is None else frames.parse_date(args.date)
    except ValueError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        return 2

    if args.json:
        print(json.dumps(observation_to_dict(body, jd), indent=2))
        return 0

    if args.info:
        return cmd_info(body)

    determined = bool(args.observations or args.determine)

    # Primary action: approaches / ephemeris / single report. After a fresh
    # determination, skip the default report (already shown) unless the user
    # asked for a specific date or visualization.
    if args.approaches is not None:
        rc = cmd_approaches(body, args.approaches, jd)
    elif args.span:
        try:
            span_days = parse_duration(args.span)
            step_days = parse_duration(args.step)
        except ValueError as exc:
            err_console.print(f"[red]error:[/red] {exc}")
            return 2
        rc = cmd_ephemeris(body, jd, span_days, step_days)
    elif determined and args.date is None and not args.animate:
        rc = 0
    else:
        rc = cmd_report(body, jd)

    if args.validate:
        cmd_validate(body, jd)

    if args.animate:
        from . import viz
        viz.animate_orbit(body, jd, console)

    if args.ascii or args.plot is not None or args.html is not None:
        from . import viz
        viz.handle_visualizations(body, jd, args, console)

    return rc


if __name__ == "__main__":
    sys.exit(main())
