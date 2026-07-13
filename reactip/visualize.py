"""Publication-quality 3D ball-and-stick rendering and reaction-path plots.

This module is the rendering layer for ReactIP sampled searches. It is kept
separate from the SE-GSM computation so figures can be regenerated from raw
outputs (``candidate_search_summary.json`` plus per-candidate trajectory files)
without recomputing anything.

Design choices:

- Structures are drawn as true 3D ball-and-stick using an orthographic
  projection of the real Cartesian coordinates (no 2D chemical-diagram
  abstraction). Atoms are shaded spheres with CPK colors and covalent radii;
  bonds are split half/half by element. Correct occlusion is handled with a
  painter's algorithm (draw far-to-near), which is robust for the small
  molecules in the ReactIP domain and avoids mplot3d z-ordering artifacts.
- All views across the frames of one trajectory share a fixed camera and fixed
  world extent so animations are stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# CPK-style colors and covalent radii (Angstrom) for the ReactIP element domain.
CPK_COLORS = {
    "H": "#ffffff",
    "C": "#404040",
    "N": "#2f63d6",
    "O": "#e8392a",
    "F": "#28c24a",
    "S": "#e6c12a",
    "Cl": "#37c037",
    "Br": "#b35a2a",
    "P": "#e8821a",
    "I": "#8b3fc0",
}
# Covalent radii and the bond-detection scale come from reactip.sampling (the
# single source of truth) so the sampler and every figure perceive identical
# connectivity. The renderer may draw a few elements the MLIP never sees
# (P, I), so those render-only radii are merged on top.
from reactip.sampling import (  # noqa: E402
    BOND_SCALE,
    COVALENT_RADII as _SAMPLING_RADII,
    DEFAULT_COVALENT_RADIUS as _DEFAULT_RADIUS,
)

COVALENT_RADII = {**_SAMPLING_RADII, "P": 1.07, "I": 1.39}
_DEFAULT_COLOR = "#9a9a9a"


@dataclass(frozen=True)
class Structure:
    symbols: tuple[str, ...]
    coords: np.ndarray  # (N, 3)


def read_xyz_frames(path: str | Path) -> list[Structure]:
    """Read all frames from a (multi-frame) XYZ file."""
    lines = Path(path).read_text().splitlines()
    frames: list[Structure] = []
    i = 0
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        try:
            n = int(lines[i].strip())
        except ValueError:
            break
        if i + 1 + n >= len(lines):
            break
        symbols: list[str] = []
        coords: list[list[float]] = []
        ok = True
        for line in lines[i + 2 : i + 2 + n]:
            parts = line.split()
            if len(parts) < 4:
                ok = False
                break
            try:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                ok = False
                break
            symbols.append(parts[0])
        if ok and len(symbols) == n:
            frames.append(Structure(tuple(symbols), np.asarray(coords, dtype=float)))
        i += n + 2
    return frames


def infer_bonds(structure: Structure, scale: float = BOND_SCALE) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    coords = structure.coords
    syms = structure.symbols
    for i in range(len(syms)):
        ri = COVALENT_RADII.get(syms[i], _DEFAULT_RADIUS)
        for j in range(i + 1, len(syms)):
            rj = COVALENT_RADII.get(syms[j], _DEFAULT_RADIUS)
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if 0.4 < d <= scale * (ri + rj):
                bonds.append((i, j))
    return bonds


def _rotation_matrix(elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)
    # Rotate about z (azimuth) then about x (elevation).
    cz, sz = np.cos(azim), np.sin(azim)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    cx, sx = np.cos(elev), np.sin(elev)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rx @ rz


def _ball(ax, x, y, radius, color, *, zorder, shade: bool = False):
    """Draw one atom as a circle: flat CPK color by default, shaded on request."""
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle

    rgb = np.array(mcolors.to_rgb(color))
    rim = tuple(rgb * 0.55)
    ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor=rim,
                         linewidth=0.9, zorder=zorder))
    if shade:
        # Two inner highlights offset toward the (upper-left) light source.
        for frac, light, off in ((0.62, 0.45, 0.30), (0.30, 0.85, 0.45)):
            hl = tuple(rgb + (1.0 - rgb) * light)
            ax.add_patch(Circle((x - off * radius, y + off * radius), radius * frac,
                                 facecolor=hl, edgecolor="none", zorder=zorder + 0.01))


def render_structure(
    ax,
    structure: Structure,
    *,
    elev: float = 18.0,
    azim: float = 32.0,
    bond_scale: float = BOND_SCALE,
    world_radius: float | None = None,
    center: np.ndarray | None = None,
    atom_scale: float = 0.42,
    highlight_atoms: Sequence[int] | None = None,
    shade: bool = False,
) -> None:
    """Render one structure as 3D ball-and-stick onto a 2D matplotlib Axes."""
    syms = structure.symbols
    rot = _rotation_matrix(elev, azim)
    if center is None:
        center = structure.coords.mean(axis=0)
    rc = (structure.coords - center) @ rot.T
    xy = rc[:, :2]
    depth = rc[:, 2]
    order = np.argsort(depth)  # far (small z) first
    bonds = infer_bonds(structure, scale=bond_scale)

    highlight = set(highlight_atoms or [])

    # Interleave bonds and atoms by depth so occlusion is approximately correct.
    drawables: list[tuple[float, str, tuple]] = []
    for i, j in bonds:
        drawables.append((min(depth[i], depth[j]) - 0.01, "bond", (i, j)))
    for idx in range(len(syms)):
        drawables.append((depth[idx], "atom", (idx,)))
    drawables.sort(key=lambda d: d[0])

    base_z = 2.0
    for k, (_, kind, payload) in enumerate(drawables):
        z = base_z + k * 0.01
        if kind == "bond":
            i, j = payload
            mid = (xy[i] + xy[j]) / 2.0
            for a, b in ((i, mid), (j, mid)):
                pt = xy[a] if isinstance(a, (int, np.integer)) else a
                ax.plot([pt[0], b[0]], [pt[1], b[1]],
                        color="#3a3a3a", lw=3.0, solid_capstyle="round",
                        zorder=z, alpha=0.95)
        else:
            (idx,) = payload
            radius = atom_scale * COVALENT_RADII.get(syms[idx], _DEFAULT_RADIUS)
            color = CPK_COLORS.get(syms[idx], _DEFAULT_COLOR)
            _ball(ax, xy[idx, 0], xy[idx, 1], radius, color, zorder=z, shade=shade)
            if idx in highlight:
                from matplotlib.patches import Circle
                ax.add_patch(Circle((xy[idx, 0], xy[idx, 1]), radius * 1.35,
                                    facecolor="none", edgecolor="#f5a623",
                                    linewidth=2.0, zorder=z + 0.5))

    if world_radius is None:
        span = float(np.max(np.ptp(rc[:, :2], axis=0))) if len(syms) > 1 else 2.0
        world_radius = span / 2.0 + 1.0
    ax.set_xlim(-world_radius, world_radius)
    ax.set_ylim(-world_radius, world_radius)
    ax.set_aspect("equal")
    ax.axis("off")


def world_extent(structures: Sequence[Structure]) -> tuple[np.ndarray, float]:
    """Shared center and radius covering all structures (for stable animation)."""
    all_coords = np.vstack([s.coords for s in structures])
    center = all_coords.mean(axis=0)
    radius = float(np.max(np.linalg.norm(all_coords - center, axis=1))) + 1.0
    return center, radius


# --------------------------------------------------------------------------
# Style helpers
# --------------------------------------------------------------------------

_ACCENT = "#e07b1a"
_TS_COLOR = "#c0392b"
_INK = "#222831"
_GRID = "#dfe3e8"


def _style_axes(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b8bec6")
    ax.tick_params(colors="#5a6470", labelsize=9)
    ax.grid(True, color=_GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def _savefig(fig, out_path: str | Path, formats: Sequence[str]) -> list[Path]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        target = out_path.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=190, bbox_inches="tight", facecolor="white")
        written.append(target)
    return written


# --------------------------------------------------------------------------
# Energy profile along an SE-GSM string
# --------------------------------------------------------------------------

def plot_energy_profile(
    energies: Sequence[float | None],
    out_path: str | Path,
    *,
    title: str = "",
    ts_node: int | None = None,
    reactant_node: int | None = None,
    product_node: int | None = None,
    formats: Sequence[str] = ("png", "pdf"),
) -> list[Path]:
    """Relative-energy profile (kcal/mol) along the string nodes."""
    import matplotlib.pyplot as plt

    idx = [i for i, e in enumerate(energies) if e is not None]
    vals = [float(energies[i]) for i in idx]
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=190)
    _style_axes(ax)
    if vals:
        ax.plot(idx, vals, "-", color=_INK, lw=1.8, zorder=2)
        ax.scatter(idx, vals, s=26, color=_INK, zorder=3,
                   edgecolors="white", linewidths=0.6)
        if ts_node is not None and ts_node < len(energies) and energies[ts_node] is not None:
            ax.scatter([ts_node], [energies[ts_node]], s=130, color=_TS_COLOR,
                       zorder=5, edgecolors="white", linewidths=1.2, label="TS")
            ax.annotate("TS", (ts_node, energies[ts_node]), textcoords="offset points",
                        xytext=(0, 9), ha="center", color=_TS_COLOR, fontweight="bold", fontsize=10)
        for node, lab, col in ((reactant_node, "reactant", "#2f63d6"),
                               (product_node, "product", "#1b9e57")):
            if node is not None and node < len(energies) and energies[node] is not None:
                ax.scatter([node], [energies[node]], s=90, color=col, zorder=4,
                           edgecolors="white", linewidths=1.0)
                ax.annotate(lab, (node, energies[node]), textcoords="offset points",
                            xytext=(0, -14), ha="center", color=col, fontsize=8)
    ax.set_xlabel("SE-GSM string node", fontsize=10)
    ax.set_ylabel("Relative energy (kcal/mol)", fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, color=_INK)
    fig.tight_layout()
    written = _savefig(fig, out_path, formats)
    plt.close(fig)
    return written


# --------------------------------------------------------------------------
# Relative-population distribution
# --------------------------------------------------------------------------

def plot_population_distribution(
    rows: Sequence[dict],
    out_path: str | Path,
    *,
    title: str = "Relative product populations",
    score_label: str = "cumulative dE (kcal/mol)",
    formats: Sequence[str] = ("png", "pdf"),
) -> list[Path]:
    """Horizontal bars of Boltzmann populations with the driving coordinates."""
    import matplotlib.pyplot as plt

    rows = list(rows)
    labels, pops, scores = [], [], []
    for r in rows:
        # Prefer a human-readable reaction label when the caller supplies one.
        coords = r.get("reaction_label") or "; ".join(r.get("driving_coords") or [])
        labels.append(f"{r.get('candidate_id')}  ({coords})")
        pops.append(100.0 * float(r.get("relative_population") or 0.0))
        scores.append(r.get("ranking_score_delta_e"))
    n = max(1, len(rows))
    fig, ax = plt.subplots(figsize=(7.4, 0.55 * n + 1.4), dpi=190)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(n)
    bars = ax.barh(y, pops or [0], color=_ACCENT, edgecolor="#9a5410", height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Relative population (%)", fontsize=10)
    ax.set_xlim(0, max(pops + [1.0]) * 1.18)
    for i, (bar, score) in enumerate(zip(bars, scores)):
        txt = f"{pops[i]:.1f}%"
        if isinstance(score, (int, float)):
            txt += f"   (dE={score:.1f})"
        ax.text(bar.get_width() + max(pops + [1.0]) * 0.02, i, txt,
                va="center", fontsize=8, color=_INK)
    ax.set_title(f"{title}\n{score_label}", fontsize=11, color=_INK)
    fig.tight_layout()
    written = _savefig(fig, out_path, formats)
    plt.close(fig)
    return written


# NOTE: a second render_trajectory_gif() used to live here (3D ball-and-stick +
# synchronized energy profile). It was dead code — never imported anywhere, and
# its (frames, energies, out_path) signature clashed with the ACTIVE
# reactip.utils.render_trajectory_gif(trajectory, output_path, ...) that
# reporting.py uses. Removed to eliminate the ambiguity (bug B1).


# --------------------------------------------------------------------------
# Multi-step reaction-path energy diagram (levels + TS peaks)
# --------------------------------------------------------------------------

def plot_reaction_path_diagram(
    steps: Sequence[dict],
    out_path: str | Path,
    *,
    title: str = "",
    formats: Sequence[str] = ("png", "pdf"),
) -> list[Path]:
    """Classic reaction-coordinate diagram for a multi-step pathway.

    ``steps`` is an ordered list, one entry per successful reaction step:
    ``label`` (e.g. candidate id or readable reaction), ``product_level``
    (cumulative dE of the product vs the root reactant, kcal/mol) and
    optionally ``ts_level`` (cumulative energy of that step's TS). The
    reactant level at 0 is implicit. Levels are drawn as horizontal bars
    connected by dashed lines, with TS peaks in between when available.
    """
    import matplotlib.pyplot as plt

    steps = list(steps)
    if not steps:
        return []

    # x layout: each species level occupies one slot; TS sits between slots.
    half = 0.30
    levels: list[tuple[float, float, str]] = [(0.0, 0.0, "reactant")]
    peaks: list[tuple[float, float, str]] = []
    for i, step in enumerate(steps, start=1):
        product_level = step.get("product_level")
        if product_level is None:
            continue
        levels.append((float(i), float(product_level), str(step.get("label", f"P{i}"))))
        ts_level = step.get("ts_level")
        if ts_level is not None:
            peaks.append((i - 0.5, float(ts_level), f"TS{i}"))

    fig, ax = plt.subplots(figsize=(2.6 * len(levels) + 1.5, 4.4), dpi=190)
    _style_axes(ax)
    ax.grid(axis="x", visible=False)

    for x, e, _ in levels:
        ax.plot([x - half, x + half], [e, e], color=_INK, lw=3.2, solid_capstyle="round", zorder=3)
    for x, e, label in peaks:
        ax.scatter([x], [e], s=46, color=_TS_COLOR, zorder=4, edgecolors="white", linewidths=0.8)
        ax.annotate(f"{label}\n{e:.1f}", (x, e), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=_TS_COLOR, fontweight="bold")

    # Connect: level -> TS peak -> next level (dashed), or level -> level.
    peak_by_x = {round(x, 3): (x, e) for x, e, _ in peaks}
    for i in range(len(levels) - 1):
        x0, e0, _ = levels[i]
        x1, e1, _ = levels[i + 1]
        mid = peak_by_x.get(round((x0 + x1) / 2.0, 3))
        if mid is not None:
            ax.plot([x0 + half, mid[0]], [e0, mid[1]], ls="--", lw=1.2, color="#8a93a0", zorder=2)
            ax.plot([mid[0], x1 - half], [mid[1], e1], ls="--", lw=1.2, color="#8a93a0", zorder=2)
        else:
            ax.plot([x0 + half, x1 - half], [e0, e1], ls="--", lw=1.2, color="#8a93a0", zorder=2)

    for x, e, label in levels:
        ax.annotate(f"{label}\n{e:.1f}", (x, e), textcoords="offset points", xytext=(0, -22),
                    ha="center", fontsize=8.5, color=_INK)

    ax.set_xticks([])
    ax.set_xlabel("Reaction progress", fontsize=10)
    ax.set_ylabel("Energy vs reactant (kcal/mol)", fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, color=_INK)
    # Pad y-limits so the below-level labels stay inside the axes.
    all_e = [e for _, e, _ in levels] + [e for _, e, _ in peaks]
    span = max(all_e) - min(all_e) if all_e else 1.0
    pad = max(0.15 * span, 4.0)
    ax.set_ylim(min(all_e) - pad - 0.10 * span, max(all_e) + pad)
    fig.tight_layout()
    written = _savefig(fig, out_path, formats)
    plt.close(fig)
    return written


# --------------------------------------------------------------------------
# Static reactant -> product 2D depiction (what changed in the reaction)
# --------------------------------------------------------------------------
#
# The trajectory GIF animates the geometry but makes it hard to read *what bond
# changed*. This renders a single static figure: the reactant with its broken
# bonds highlighted (red) next to the product with its formed bonds highlighted
# (green), as flat 2D chemical structures via RDKit. It falls back to 3D
# ball-and-stick panels when RDKit is unavailable or a geometry can't be
# perceived (e.g. exploded failed runs). Kept separate from the GIF on purpose:
# static vs animated are different media, and this file is independently useful.

_BROKEN_COLOR = "#d6382c"  # red   — bonds present in reactant, gone in product
_FORMED_COLOR = "#1c9f56"  # green — bonds absent in reactant, present in product


def bond_change(
    reactant: Structure, product: Structure, *, scale: float = 1.25
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Net connectivity change between two structures with a shared atom order.

    Returns ``(formed, broken)`` as sorted lists of 0-based ``(i, j)`` index
    pairs, using the same distance criterion as the ball-and-stick renderer.
    A null reaction (product ~ reactant) yields two empty lists.
    """
    rb = {tuple(sorted(b)) for b in infer_bonds(reactant, scale=scale)}
    pb = {tuple(sorted(b)) for b in infer_bonds(product, scale=scale)}
    return sorted(pb - rb), sorted(rb - pb)


def _hex_to_rgb01(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


# Sentinel atom-map number used to shield specific H atoms from RemoveHs
# (RemoveHs with removeMapped=False keeps mapped atoms); stripped before drawing.
_KEEP_MAPNUM = 9009


def _condense_hs(mol, keep_atoms: Sequence[int] = ()):
    """Return ``(mol_no_H, old2new)`` — collapse C–H hydrogens to implicit for a
    clean skeletal depiction while keeping the chemistry readable.

    Heteroatom hydrogens survive as element labels (``OH``/``NH``); any H whose
    index is in ``keep_atoms`` (0-based) is also retained — used to keep an H
    that sits on a highlighted formed/broken bond so H-transfer steps still show
    the change. ``old2new`` maps each original atom index to its index in the
    returned mol (removed hydrogens are absent from the map). Stereochemistry and
    implicit-H counts are handled by RDKit's ``RemoveHs``, so wedge bonds and CIP
    labels stay correct. On any failure the input mol is returned with an
    identity map, so drawing degrades to the (cluttered but valid) all-H view.
    """
    from rdkit import Chem

    keep = {int(i) for i in keep_atoms}
    work = Chem.Mol(mol)
    for a in work.GetAtoms():
        a.SetIntProp("_ridx", a.GetIdx())
    for i in keep:
        if 0 <= i < work.GetNumAtoms() and work.GetAtomWithIdx(i).GetAtomicNum() == 1:
            work.GetAtomWithIdx(i).SetAtomMapNum(_KEEP_MAPNUM)

    params = Chem.RemoveHsParameters()
    params.removeMapped = False  # protect the H atoms we tagged above
    out = None
    for sanitize in (True, False):  # graph-mode (unsanitized) mols need sanitize=False
        try:
            out = Chem.RemoveHs(work, params, sanitize=sanitize)
            break
        except Exception:
            continue
    if out is None:
        return mol, {i: i for i in range(mol.GetNumAtoms())}

    for a in out.GetAtoms():  # strip the protection sentinel so it isn't drawn
        if a.GetAtomMapNum() == _KEEP_MAPNUM:
            a.SetAtomMapNum(0)
    old2new = {a.GetIntProp("_ridx"): a.GetIdx()
               for a in out.GetAtoms() if a.HasProp("_ridx")}
    return out, old2new


def _layout_2d(mol) -> None:
    """Assign 2D coordinates in place: CoordGen (Schrödinger's algorithm — clean
    zig-zag chains and well-proportioned rings) with a fallback to the classic
    RDKit depictor. Any pre-existing (e.g. 3D) conformer is cleared first."""
    from rdkit.Chem import rdCoordGen, rdDepictor

    mol.RemoveAllConformers()
    try:
        rdCoordGen.AddCoords(mol)
    except Exception:
        try:
            rdDepictor.Compute2DCoords(mol)
        except Exception:
            pass


def _clean_perception(mol) -> bool:
    """True when a full bond-order perception is confident enough to draw as a
    Lewis structure: no atom carries a formal charge or a radical electron.

    ``DetermineBonds`` forces charge-separated (C⁺/C⁻, O⁺/O⁻) or radical
    structures onto the distorted / half-broken SE-GSM geometries it can't fit
    with clean octets. Those decorations are perception artifacts, not chemistry,
    so such a molecule is better shown as a connectivity-only skeleton — matching
    this module's two-tier "full only when confident, else graph" design."""
    for a in mol.GetAtoms():
        if a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() != 0:
            return False
    return True


def _prep_graph_skeleton(mol) -> None:
    """In place, turn a connectivity-only mol into a clean single-bond skeleton:
    all bonds single (no perceived orders/aromaticity), no formal charges, no
    radicals, and no implicit hydrogens. Drawn *without* a sanitize/prepare pass
    (see the callers), this yields a bare connectivity skeleton — heavy-atom
    vertices joined by single lines — with no spurious ``OH``/``·`` decorations,
    e.g. spectator O2 shows as ``O–O`` rather than a fake ``HO–OH`` or ``·O–O·``.
    Used for the low-confidence graph-mode fallback depiction."""
    from rdkit import Chem

    for b in mol.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in mol.GetAtoms():
        a.SetFormalCharge(0)
        a.SetNumRadicalElectrons(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)  # don't invent implicit H (no fake OH on O/N vertices)
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    except Exception:
        pass


def _mol_2d_png(
    structure: Structure,
    highlight_ij: Sequence[tuple[int, int]],
    hex_color: str,
    *,
    size: tuple[int, int] = (560, 470),
) -> bytes | None:
    """Perceive bonds from 3D coords, lay the molecule out in 2D, and draw it as
    a chemical structure with ``highlight_ij`` bonds/atoms colored. Returns PNG
    bytes, or ``None`` if RDKit is missing or perception fails."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        from rdkit.Chem.Draw import rdMolDraw2D
        from rdkit.Geometry import Point3D
    except Exception:
        return None

    def _build():
        rw = Chem.RWMol()
        conf = Chem.Conformer(len(structure.symbols))
        for i, (s, xyz) in enumerate(zip(structure.symbols, structure.coords)):
            rw.AddAtom(Chem.Atom(str(s)))
            conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
        m = rw.GetMol()
        m.AddConformer(conf, assignId=True)
        return m

    # Prefer full bond-order perception (nice double bonds / aromatics), but only
    # trust it when it comes out clean (no forced charges/radicals); otherwise
    # fall back to a connectivity-only skeleton, which tolerates odd valences.
    mol = _build()
    graph_mode = False
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        if not _clean_perception(mol):
            raise ValueError("charge/radical-separated perception -> use graph")
    except Exception:
        graph_mode = True
        mol = _build()
        try:
            rdDetermineBonds.DetermineConnectivity(mol)
        except Exception:
            return None

    # Collapse C–H to implicit hydrogens for a clean skeletal depiction, keeping
    # any H that is an endpoint of a highlighted (formed/broken) bond so the
    # change stays visible; remap the highlight indices onto the condensed mol.
    keep_h = {int(a) for pair in highlight_ij for a in pair
              if 0 <= int(a) < mol.GetNumAtoms()
              and mol.GetAtomWithIdx(int(a)).GetAtomicNum() == 1}
    mol, _old2new = _condense_hs(mol, keep_h)
    highlight_ij = [(_old2new[i], _old2new[j]) for (i, j) in highlight_ij
                    if i in _old2new and j in _old2new]

    # Graph-mode: render as a clean single-bond skeleton (no orders/charges/dots).
    if graph_mode:
        _prep_graph_skeleton(mol)

    # Drop the 3D conformer and lay the molecule out in 2D (CoordGen).
    _layout_2d(mol)

    hb: list[int] = []
    ha: set[int] = set()
    for i, j in highlight_ij:
        b = mol.GetBondBetweenAtoms(int(i), int(j))
        if b is not None:
            hb.append(b.GetIdx())
        ha.update((int(i), int(j)))
    color = _hex_to_rgb01(hex_color)
    acolors = {a: color for a in ha}
    bcolors = {b: color for b in hb}

    def _fresh_drawer():
        dd = rdMolDraw2D.MolDraw2DCairo(*size)
        o = dd.drawOptions()
        o.bondLineWidth = 2
        o.padding = 0.10
        o.highlightBondWidthMultiplier = 14
        return dd, o

    # Full-perception (clean) mols: prepare (sanitize/kekulize) + draw for nice
    # double bonds / aromatic rings. Graph-mode skeletons skip prepare (handled by
    # the no-prepare path below) so sanitize doesn't re-derive the radicals/charges
    # that _prep_graph_skeleton deliberately stripped.
    if not graph_mode:
        d, _ = _fresh_drawer()
        try:
            rdMolDraw2D.PrepareAndDrawMolecule(
                d, mol, highlightAtoms=list(ha), highlightBonds=hb,
                highlightAtomColors=acolors, highlightBondColors=bcolors,
            )
            d.FinishDrawing()
            return d.GetDrawingText()
        except Exception:
            pass

    # Fallback for connectivity-only / hypervalent mols that can't be kekulized
    # (e.g. collapsed blow-up geometries): draw without prepare on a FRESH drawer
    # — reusing the prepared drawer, or degenerate 2D coords from Compute2DCoords,
    # crashes the Cairo backend ("Cannot normalize a zero length vector"). If the
    # first attempt still crashes, re-lay-out with CoordGen (more robust) and retry
    # so the product renders in 2D rather than falling back to 3D ball-and-stick.
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    for relayout in (False, True):
        if relayout:
            try:
                from rdkit.Chem import rdCoordGen

                mol.RemoveAllConformers()
                rdCoordGen.AddCoords(mol)
                mol.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(mol)
            except Exception:
                continue
        d, o = _fresh_drawer()
        o.prepareMolsBeforeDrawing = False
        try:
            d.DrawMolecule(
                mol, highlightAtoms=list(ha), highlightBonds=hb,
                highlightAtomColors=acolors, highlightBondColors=bcolors,
            )
            d.FinishDrawing()
            return d.GetDrawingText()
        except Exception:
            continue
    return None


# Sentinel distinguishing "PNG not supplied, compute it" from "PNG is None,
# perception failed -> use the 3D fallback" in draw_reaction_change_panels.
_UNSET = object()


def draw_reaction_change_panels(
    ax_r,
    ax_mid,
    ax_p,
    reactant: Structure,
    product: Structure,
    formed: Sequence[tuple[int, int]],
    broken: Sequence[tuple[int, int]],
    *,
    r_png: bytes | None = _UNSET,  # type: ignore[assignment]
    p_png: bytes | None = _UNSET,  # type: ignore[assignment]
) -> None:
    """Draw the reactant → product depiction onto three caller-provided axes.

    ``ax_r``/``ax_p`` get the reactant/product panels (broken bonds red on the
    reactant, formed bonds green on the product); ``ax_mid`` gets the arrow.
    Shared by :func:`render_reaction_change` (static PNG) and the ``reaction.gif``
    compositor so both read identically.

    ``r_png``/``p_png`` may be pre-rendered :func:`_mol_2d_png` bytes (or ``None``
    to force the 3D ball-and-stick fallback for that panel); left unset, they are
    computed here. Passing cached bytes lets an animation redraw this static panel
    on every frame without repeating the (expensive) RDKit perception.
    """
    import io

    import matplotlib.pyplot as plt

    if r_png is _UNSET:
        r_png = _mol_2d_png(reactant, broken, _BROKEN_COLOR)
    if p_png is _UNSET:
        p_png = _mol_2d_png(product, formed, _FORMED_COLOR)

    for ax, lab, png, struct, hl, col in (
        (ax_r, "Reactant", r_png, reactant, broken, _BROKEN_COLOR),
        (ax_p, "Product", p_png, product, formed, _FORMED_COLOR),
    ):
        if png is not None:
            ax.imshow(plt.imread(io.BytesIO(png), format="png"))
        else:  # RDKit unavailable / perception failed -> 3D ball-and-stick
            hi = sorted({i for pair in hl for i in pair})
            render_structure(ax, struct, highlight_atoms=hi)
        ax.set_title(lab, fontsize=12.5, color=col, fontweight="bold")
        ax.axis("off")

    ax_mid.axis("off")
    ax_mid.text(0.5, 0.5, "→", ha="center", va="center", fontsize=34,
                color=_ACCENT, transform=ax_mid.transAxes)


def render_reaction_change(
    reactant: Structure,
    product: Structure,
    out_path: str | Path,
    *,
    title: str = "",
    subtitle: str = "",
    formed: Sequence[tuple[int, int]] | None = None,
    broken: Sequence[tuple[int, int]] | None = None,
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """Static reactant -> product figure with the changed bonds highlighted.

    Broken bonds are drawn red on the reactant panel, formed bonds green on the
    product panel. ``formed``/``broken`` default to :func:`bond_change`. Uses
    RDKit 2D depictions, falling back per-panel to 3D ball-and-stick when a
    structure can't be perceived.
    """
    import matplotlib.pyplot as plt

    if formed is None or broken is None:
        f2, b2 = bond_change(reactant, product)
        formed = f2 if formed is None else formed
        broken = b2 if broken is None else broken

    fig = plt.figure(figsize=(10.4, 5.3), dpi=190)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 0.16, 1.0), wspace=0.02)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[0, 2])

    draw_reaction_change_panels(ax_r, ax_mid, ax_p, reactant, product, formed, broken)

    sup = title + (f"\n{subtitle}" if subtitle else "")
    if sup.strip():
        fig.suptitle(sup, fontsize=11, color=_INK, y=0.99)
    # subplots_adjust (not tight_layout) avoids an imshow-incompatibility warning;
    # _savefig crops to content with bbox_inches="tight".
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.02)
    written = _savefig(fig, out_path, formats)
    plt.close(fig)
    return written
