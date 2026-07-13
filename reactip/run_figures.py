"""Per-run figure helpers: render ``reaction.png`` and ``trajectory.gif`` for a
single SE-GSM run directory straight from its raw outputs (grown-string XYZ +
``summary.json``), with no SE-GSM recomputation.

One implementation, shared by three callers:

- ``run_se_gsm.run_with_reporting`` — the single driving-coordinate mode, so a
  one-off run produces ``reaction.png`` and ``trajectory.gif`` by default (the
  sampled-search report already produced both per candidate);
- ``scripts/make_reaction_pngs.py`` / ``scripts/make_trajectory_gifs.py`` — the
  standalone regenerators that sweep a runs tree and (re)build missing figures;
- indirectly, anything else that has a run directory on disk.

Trajectory resolution goes through :func:`reactip.utils.resolve_trajectory_source`,
so a stale absolute ``trajectory_source`` recorded on the compute node still
finds the copied-over XYZ. Reactant/product frames come from the summary's
``reactant_node`` / ``product_node`` (falling back to first/last frame), so the
depiction matches the animation and a "converged" run that did not actually
react is shown honestly as a null reaction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .utils import (
    parse_molden_xyz_trajectory,
    render_trajectory_gif,
    resolve_trajectory_source,
)


def _labels():
    """(describe_reaction, short_status), lazily imported to avoid a hard
    dependency on the heavier reporting module; falls back to plain formatters."""
    try:
        from .reporting import _short_status, describe_reaction

        return describe_reaction, _short_status
    except Exception:  # pragma: no cover
        return (lambda dc, symbols=None: "; ".join(dc or [])), (lambda s: str(s))


def _load_summary(run_dir: Path) -> dict:
    p = run_dir / "summary.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _bond_str(pairs, symbols) -> str:
    return ", ".join(f"{symbols[i]}{i + 1}–{symbols[j]}{j + 1}" for i, j in pairs)


def _run_id(summary: dict, default: int) -> int:
    return (summary.get("se_gsm_parameters") or {}).get("run_id", default)


def _reaction_context(run_dir: Path, summ: dict, run_id: int) -> dict | None:
    """Shared reactant → product data for ``reaction.png`` and ``reaction.gif``.

    Resolves the trajectory locally (robust to stale ``trajectory_source``), picks
    the reactant/product frames from the summary's ``reactant_node`` /
    ``product_node`` (falling back to first/last frame, so a "converged" run that
    did not react shows honestly as a null reaction), and derives the changed
    bonds plus a human-readable ``change`` / ``dE`` / ``status``. Returns ``None``
    when no trajectory or frames are found.
    """
    from .visualize import Structure, bond_change

    src = resolve_trajectory_source(run_dir, summ.get("trajectory_source"), _run_id(summ, run_id))
    if src is None:
        return None
    traj = parse_molden_xyz_trajectory(src)
    if not traj.frames:
        return None

    frames = traj.frames
    n = len(frames)
    rn, pn = summ.get("reactant_node"), summ.get("product_node")
    rf = frames[rn] if isinstance(rn, int) and 0 <= rn < n else frames[0]
    pf = frames[pn] if isinstance(pn, int) and 0 <= pn < n else frames[-1]
    reactant = Structure(tuple(rf.symbols), np.asarray(rf.coordinates, float))
    product = Structure(tuple(pf.symbols), np.asarray(pf.coordinates, float))

    formed, broken = bond_change(reactant, product)
    syms = reactant.symbols
    parts = []
    if broken:
        parts.append("break " + _bond_str(broken, syms))
    if formed:
        parts.append("form " + _bond_str(formed, syms))
    change = " · ".join(parts) if parts else "no net bond change (null reaction)"

    dE = summ.get("delta_e")
    if not isinstance(dE, (int, float)):
        dE = summ.get("product_delta_e")

    return {
        "traj": traj,
        "reactant": reactant,
        "product": product,
        "formed": formed,
        "broken": broken,
        "change": change,
        "dE": dE,
        "status": summ.get("status"),
    }


def _change_line(change: str, dE, status, short_status) -> str:
    """The bond-change / ΔE / status line shared by reaction.png (its 3rd title
    line) and reaction.gif (its 2nd title line)."""
    line = change + (f"    ΔE = {dE:.1f} kcal/mol" if isinstance(dE, (int, float)) else "")
    if status is not None:
        line += f"    [{short_status(status)}]"
    return line


def render_run_reaction_png(
    run_dir: str | Path,
    *,
    summary: dict | None = None,
    run_id: int = 0,
    overwrite: bool = False,
) -> Path | None:
    """Write ``<run_dir>/reaction.png`` (reactant → product, changed bonds
    highlighted). Returns the path, or None when no trajectory/frames are found.

    ``summary`` may be passed directly (single-run mode, before summary.json is
    on disk); otherwise it is read from ``run_dir/summary.json``.
    """
    from .visualize import render_reaction_change

    _describe, short_status = _labels()
    run_dir = Path(run_dir)
    out = run_dir / "reaction.png"
    if out.exists() and not overwrite:
        return out

    summ = summary if summary is not None else _load_summary(run_dir)
    ctx = _reaction_context(run_dir, summ, run_id)
    if ctx is None:
        return None

    line2 = _change_line(ctx["change"], ctx["dE"], ctx["status"], short_status)
    render_reaction_change(
        ctx["reactant"], ctx["product"], out,
        title=summ.get("case") or run_dir.name,
        subtitle=f"{summ.get('reaction_label', '')}\n{line2}",
        formed=ctx["formed"], broken=ctx["broken"],
    )
    return out


def render_run_trajectory_gif(
    run_dir: str | Path,
    *,
    summary: dict | None = None,
    run_id: int = 0,
    overwrite: bool = False,
    duration: float = 1.0,
    title: str | None = None,
) -> Path | None:
    """Write ``<run_dir>/trajectory.gif`` (3D ball-and-stick + synchronized
    energy panel). Returns the path, or None when no trajectory/frames are found.

    Raises ``ModuleNotFoundError`` when the optional plotting deps (imageio) are
    absent, so callers can fall back or report the missing dependency.
    """
    describe_reaction, short_status = _labels()
    run_dir = Path(run_dir)
    out = run_dir / "trajectory.gif"
    if out.exists() and not overwrite:
        return out

    summ = summary if summary is not None else _load_summary(run_dir)
    src = resolve_trajectory_source(run_dir, summ.get("trajectory_source"), _run_id(summ, run_id))
    if src is None:
        return None
    traj = parse_molden_xyz_trajectory(src)
    if not traj.frames:
        return None

    if title is None:
        symbols = traj.frames[0].symbols
        dc = summ.get("driving_coords")
        label = describe_reaction(dc, symbols) if dc else summ.get("reaction_label", "")
        title = summ.get("case") or run_dir.name
        if label:
            title = f"{title}  ({label})"
        status = summ.get("status")
        if status is not None:
            title += f"  [{short_status(status)}]"
    return render_trajectory_gif(traj, out, title=title, duration=duration)


def _draw_energy_panel(ax, traj, frame_index, *, indices, values, limits) -> None:
    """Static relative-energy profile along the string with a red marker on the
    current frame. The curve, axes and limits are fixed across frames (only the
    marker moves), matching the energy panel baked into ``trajectory.gif``.
    """
    ax.set_facecolor("white")
    ax.set_title("Relative energy along string", fontsize=10, pad=8)
    ax.set_xlabel("String node", fontsize=9)
    ax.set_ylabel("Rel. E [kcal/mol]", fontsize=9)
    ax.tick_params(labelsize=8, colors="#475569")
    ax.axhline(0.0, color="#cbd5e1", linewidth=1.0, linestyle="--")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    if len(indices) > 0:
        ax.plot(indices, values, color="#475569", linewidth=1.8, marker="o",
                markersize=4.2, markerfacecolor="white", markeredgecolor="#475569")
        e = traj.frames[frame_index].energy_kcal_mol
        if e is not None:
            ax.scatter([frame_index + 1], [e], s=72, color="#dc2626",
                       edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_xlim(0.7, len(traj.frames) + 0.3)
    ax.set_ylim(*limits)
    ax.grid(True, linestyle=":", linewidth=0.7, color="#e2e8f0")


def render_run_reaction_gif(
    run_dir: str | Path,
    *,
    summary: dict | None = None,
    run_id: int = 0,
    overwrite: bool = False,
    duration: float = 1.0,
    elev: float | None = None,
    azim: float | None = None,
    show_labels: bool = True,
    label_h: bool = False,
) -> Path | None:
    """Write ``<run_dir>/reaction.gif`` — a single three-panel animation built
    from the same raw outputs as ``reaction.png`` and ``trajectory.gif`` (no
    SE-GSM recomputation):

    - top-left  — the reaction *movie* (animated 3D ball-and-stick, fixed camera);
    - top-right — the energy profile (static curve; a marker tracks the frame);
    - bottom    — the static reactant → product depiction (changed bonds
      highlighted), full width.

    The title reproduces ``reaction.png``'s last two lines (the reaction label,
    then the bond change / ΔE / status), deliberately dropping ``reaction.png``'s
    top case-name line. ``trajectory.gif``'s own title is not used.

    Returns the path, or ``None`` when no trajectory/frames are found. Raises
    ``ModuleNotFoundError`` when the optional plotting deps (imageio) are absent,
    so callers can fall back or report the missing dependency.
    """
    try:
        import imageio.v2 as imageio
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "reaction.gif export requires the optional plotting dependencies: "
            "matplotlib and imageio."
        ) from exc

    from .utils import _AZIM, _ELEV, _render_structure_3d
    from .visualize import (
        _BROKEN_COLOR,
        _FORMED_COLOR,
        _mol_2d_png,
        draw_reaction_change_panels,
    )

    _describe, short_status = _labels()
    run_dir = Path(run_dir)
    out = run_dir / "reaction.gif"
    if out.exists() and not overwrite:
        return out

    summ = summary if summary is not None else _load_summary(run_dir)
    ctx = _reaction_context(run_dir, summ, run_id)
    if ctx is None:
        return None

    traj = ctx["traj"]
    reactant, product = ctx["reactant"], ctx["product"]
    formed, broken = ctx["formed"], ctx["broken"]

    # Title = reaction.png's last two lines (reaction label + change/ΔE/status);
    # the top case-name line is intentionally dropped.
    line_top = summ.get("reaction_label", "") or ""
    line_bot = _change_line(ctx["change"], ctx["dE"], ctx["status"], short_status)
    title = "\n".join(s for s in (line_top, line_bot) if s.strip())

    elev = _ELEV if elev is None else elev
    azim = _AZIM if azim is None else azim

    # Fixed camera / world extent over ALL frames so the movie does not jitter.
    all_coords = np.vstack([f.coordinates for f in traj.frames])
    minima, maxima = all_coords.min(axis=0), all_coords.max(axis=0)
    center = (minima + maxima) / 2.0
    radius = max(float((maxima - minima).max()) / 2.0, 0.8) + 0.35

    # Static energy profile: curve/limits computed once; only the marker moves.
    epts = [
        (i, f.energy_kcal_mol)
        for i, f in enumerate(traj.frames)
        if f.energy_kcal_mol is not None
    ]
    if epts:
        e_idx = np.asarray([i + 1 for i, _ in epts], dtype=float)
        e_val = np.asarray([e for _, e in epts], dtype=float)
        pad = max((e_val.max() - e_val.min()) * 0.12, 8.0)
        e_lim = (float(e_val.min() - pad), float(e_val.max() + pad))
    else:
        e_idx, e_val, e_lim = np.asarray([]), np.asarray([]), (-1.0, 1.0)

    # The reactant → product panel is static: perceive the 2D depictions ONCE and
    # reuse the cached PNG bytes on every frame (RDKit perception is the slow part).
    r_png = _mol_2d_png(reactant, broken, _BROKEN_COLOR)
    p_png = _mol_2d_png(product, formed, _FORMED_COLOR)

    n = len(traj.frames)
    # Stream frames to the GIF writer one at a time. Accumulating every RGBA
    # frame in a list (this figure is ~1540x1232) is what made a high-worker
    # batch OOM; with a streaming writer peak memory is O(1) frames per worker.
    with imageio.get_writer(out, mode="I", duration=duration, loop=0) as writer:
        for i, frame in enumerate(traj.frames):
            fig = plt.figure(figsize=(11.0, 8.8), dpi=140)
            fig.patch.set_facecolor("#f8fafc")
            gs = fig.add_gridspec(
                2, 1, height_ratios=(1.18, 1.0),
                left=0.035, right=0.972, top=0.90, bottom=0.02, hspace=0.26,
            )
            top = gs[0].subgridspec(1, 2, width_ratios=(1.28, 1.0), wspace=0.14)
            ax3d = fig.add_subplot(top[0, 0], projection="3d")
            ax_e = fig.add_subplot(top[0, 1])
            bottom = gs[1].subgridspec(1, 3, width_ratios=(1.0, 0.16, 1.0), wspace=0.02)
            ax_r = fig.add_subplot(bottom[0, 0])
            ax_mid = fig.add_subplot(bottom[0, 1])
            ax_p = fig.add_subplot(bottom[0, 2])

            ax3d.set_facecolor("#f8fafc")
            _render_structure_3d(
                ax3d, frame, center=center, radius=radius, elev=elev, azim=azim,
                show_labels=show_labels, label_h=label_h,
            )
            ax3d.set_title(f"reaction movie · frame {i + 1}/{n}", fontsize=11, pad=10)

            _draw_energy_panel(ax_e, traj, i, indices=e_idx, values=e_val, limits=e_lim)

            draw_reaction_change_panels(
                ax_r, ax_mid, ax_p, reactant, product, formed, broken,
                r_png=r_png, p_png=p_png,
            )

            if title.strip():
                fig.suptitle(title, fontsize=12, color="#222831", y=0.985)

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3].copy())
            plt.close(fig)
    return out
