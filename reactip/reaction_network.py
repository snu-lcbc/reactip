"""Reaction-network figures that use molecule *structures* as nodes (RDKit 2D
depictions) connected by labelled reaction arrows.

Two views, both built straight from a sampled search's
``candidate_search_summary.json`` (no SE-GSM recomputation):

- :func:`build_onestep_network` — the root reactant in the centre with its
  one-step products (``path_depth == 1``) around it, radial layout.
- :func:`build_multistep_network` — the root reactant on the left expanding
  left-to-right through successive elementary steps to the products reached
  after 2-3 steps, layered tree layout.

Each candidate carries ``parent_id`` (tree edges), ``driving_coords`` (the
step), ``product_delta_e`` (stepwise ΔE vs parent), ``path_product_delta_e``
(cumulative ΔE vs root), ``local_ts_barrier_e`` and ``status``.

This module lives in the package (not ``scripts/``) so the default reporting
pipeline can import it; ``scripts/reaction_network.py`` is a thin re-export
shim kept for backwards compatibility. It replaces the old ball-and-stick
``render_success_pathway`` figure.
"""
from __future__ import annotations

import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib.colors import Normalize, to_hex  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from reactip.visualize import Structure, bond_change, read_xyz_frames  # noqa: E402,F401
from reactip.visualize import _mol_2d_png  # noqa: E402

_FORMED_COLOR = "#1c9f56"
_ROOT_FRAME = "#2166ac"
ROOT = "root"


def _describe_reaction(driving_coords, symbols=None) -> str:
    """Readable reaction label, delegating to reporting.describe_reaction.

    Imported lazily to avoid a circular import (reporting imports this module).
    Falls back to the raw driving-coordinate lines if reporting is unavailable.
    """
    try:
        from reactip.reporting import describe_reaction
    except Exception:  # pragma: no cover
        return "; ".join(driving_coords or [])
    return describe_reaction(driving_coords, symbols)


# --------------------------------------------------------------------------
# Data loading / path resolution
# --------------------------------------------------------------------------
def load_search(run_dir: str | Path):
    """Return (summary_dict, case_dir) from a run dir or a summary json path."""
    p = Path(run_dir)
    if p.is_dir():
        p = p / "candidate_search_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"no candidate_search_summary.json at {p}")
    return json.loads(p.read_text()), p.parent


def _resolve_dir(cand: dict, case_dir: Path) -> Path | None:
    """Candidate run directory, resolved locally (recorded abs path may be stale)."""
    rd = cand.get("run_directory")
    if not rd:
        return None
    p = Path(rd)
    if p.exists():
        return p
    parts = p.parts
    if case_dir.name in parts:  # reconstruct relative to the (possibly renamed) case dir
        rel = Path(*parts[parts.index(case_dir.name) + 1:])
        if (case_dir / rel).exists():
            return case_dir / rel
    return None


def _first_structure(path: str | Path | None) -> Structure | None:
    if not path or not Path(path).exists():
        return None
    frames = read_xyz_frames(path)
    return frames[0] if frames else None


def root_structure(summary: dict, case_dir: Path) -> Structure | None:
    """Root reactant: initial_xyz_file, else any depth-1 candidate's reactant.xyz."""
    s = _first_structure(summary.get("initial_xyz_file"))
    if s is not None:
        return s
    for c in summary.get("candidates", []):
        if c.get("path_depth") == 1:
            d = _resolve_dir(c, case_dir)
            if d and (d / "reactant.xyz").exists():
                return _first_structure(d / "reactant.xyz")
    return None


def candidate_structures(cand: dict, case_dir: Path):
    """(reactant, product) structures for a candidate (product.xyz / reactant.xyz)."""
    d = _resolve_dir(cand, case_dir)
    reactant = product = None
    if d:
        product = _first_structure(d / "product.xyz")
        reactant = _first_structure(d / "reactant.xyz")
    if product is None:
        product = _first_structure(cand.get("product_xyz"))
    if reactant is None:
        reactant = _first_structure(cand.get("reactant_xyz"))
    return reactant, product


# --------------------------------------------------------------------------
# Node rendering + labels
# --------------------------------------------------------------------------
def node_png(structure: Structure, formed_bonds=(), *, size=(430, 360)) -> bytes | None:
    """RDKit 2D depiction with formed bonds highlighted green (for a product node)."""
    if structure is None:
        return None
    return _mol_2d_png(structure, formed_bonds, _FORMED_COLOR, size=size)


def step_label(cand: dict, symbols=None) -> str:
    """Concise reaction label for an edge, e.g. 'form C3-C5, C4-O6'."""
    return _describe_reaction(cand.get("driving_coords"), symbols)


def delta_e(cand: dict, *, cumulative: bool) -> float | None:
    v = cand.get("path_product_delta_e") if cumulative else cand.get("product_delta_e")
    return v if isinstance(v, (int, float)) else None


class DeltaEColor:
    """Map ΔE (kcal/mol) to a color: green=favorable(low) -> red=unfavorable(high)."""

    def __init__(self, values, cmap="RdYlGn_r"):
        vals = [v for v in values if isinstance(v, (int, float))]
        lo = min(vals) if vals else 0.0
        hi = max(vals) if vals else 1.0
        if hi - lo < 1e-6:
            lo, hi = lo - 1.0, hi + 1.0
        self.norm = Normalize(vmin=lo, vmax=hi)
        self.cmap = cm.get_cmap(cmap)

    def hex(self, v):
        if not isinstance(v, (int, float)):
            return "#9aa3ad"
        return to_hex(self.cmap(self.norm(v)))

    def mappable(self):
        m = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        m.set_array([])
        return m


# --------------------------------------------------------------------------
# Generic network drawing (molecule images as nodes, labeled arrows as edges)
# --------------------------------------------------------------------------
def draw_network(
    nodes: list[dict],
    edges: list[dict],
    out_path: str | Path,
    *,
    title: str = "",
    subtitle: str = "",
    figsize=(13, 13),
    node_w: float = 0.15,
    colorbar: "DeltaEColor | None" = None,
    colorbar_label: str = "cumulative ΔE vs reactant (kcal/mol)",
    legend: "list | None" = None,
    legend_loc: str = "lower left",
    legend_anchor: "tuple | None" = None,
    legend_ncol: int = 1,
    legend_style: str = "line",
    title_loc: str = "center",
    edge_label_fontsize: float = 10.0,
    edge_box_pad: float = 0.36,
    edge_box_linewidth: float = 1.1,
    caption_fontsize: float = 11.5,
    node_aspect: float = 360.0 / 430.0,
    node_frame_lw: float = 2.7,
    root_frame_lw: float = 3.6,
    wrap_labels: bool = True,
    formats=("png",),
) -> list[Path]:
    """Draw a reaction network.

    ``nodes``: dicts with ``id``, ``pos`` (x,y in [0,1]), ``png`` (bytes|None),
    ``caption`` (below), ``frame_color`` (hex border), optional ``w`` (node
    width in figure fraction) and ``is_root``.
    ``edges``: dicts with ``src``, ``dst`` (node ids), ``label`` (str),
    ``color`` (hex), ``width``; optional ``rad`` (arc curvature, >0 bows to the
    left of src->dst), ``style`` (matplotlib linestyle, e.g. dashed for an
    attempted-but-unconverged step), ``label_t`` (0-1 along the chord) and
    ``label_off`` (perpendicular label nudge in chord-length units).
    ``legend``: optional list of ``(color, linestyle, label)`` tuples drawn as a
    frameless key at the lower left. ``title_loc``: ``"center"`` or ``"left"``.
    ``edge_label_fontsize`` / ``edge_box_pad`` / ``edge_box_linewidth`` size the
    reaction-label text and its rounded box; ``caption_fontsize`` sizes the node
    captions. Defaults match the one-step figure; the multi-step view passes
    smaller values so its labels don't dwarf the molecule structures.
    ``node_aspect`` is the node box's height/width (default ≈0.84 = the 430×360
    depiction aspect); *lower* makes boxes wider (render the depiction PNG at the
    same aspect so the molecule isn't stretched). ``node_frame_lw`` /
    ``root_frame_lw`` set the thickness of the colored box border (product / root).
    ``wrap_labels``: wrap each edge label after every comma onto stacked lines so
    the "break …, form …" boxes stay compact and legible when the figure is
    shrunk into a slide or report.
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.axis("off")

    fig_w, fig_h = figsize
    img_ar = node_aspect  # node box height / width (default matches the depiction)

    def node_size(n):
        w = n.get("w", node_w)
        h = w * (fig_w / fig_h) * img_ar  # preserve molecule aspect on the page
        return w, h

    pos = {n["id"]: n["pos"] for n in nodes}
    size = {n["id"]: node_size(n) for n in nodes}

    # Edges (drawn under the node images).
    for e in edges:
        if e["src"] not in pos or e["dst"] not in pos:
            continue
        (x0, y0), (x1, y1) = pos[e["src"]], pos[e["dst"]]
        # shrink each end to ~the node's half-size (in points) so arrows touch borders
        sa = 0.5 * size[e["src"]][0] * fig_w * 72 * 0.92
        sb = 0.5 * size[e["dst"]][0] * fig_w * 72 * 0.92
        rad = e.get("rad", 0.0)
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1), transform=bg.transData, arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=16, lw=e.get("width", 1.6), color=e.get("color", "#8a93a0"),
            linestyle=e.get("style", "-"),
            shrinkA=sa, shrinkB=sb, zorder=1, joinstyle="round",
        )
        bg.add_patch(arrow)
        if e.get("label"):
            t = e.get("label_t", 0.62)  # place label toward the product end
            mx, my = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            # follow the arc's bow and any explicit perpendicular nudge so the
            # label sits on the curved arrow, not across the chord
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy) or 1.0
            px, py = -dy / L, dx / L
            mx += (rad * 0.5 + e.get("label_off", 0.0)) * px * L
            my += (rad * 0.5 + e.get("label_off", 0.0)) * py * L
            label = e["label"]
            if wrap_labels:
                # break "break …, form …" onto stacked lines after each comma
                label = re.sub(r",\s*", ",\n", label)
            bg.text(mx, my, label, fontsize=edge_label_fontsize, ha="center", va="center",
                    color="#1a1f26", zorder=2,
                    bbox=dict(boxstyle=f"round,pad={edge_box_pad}", fc="white",
                              ec="#7a828c", lw=edge_box_linewidth, alpha=1.0))

    # Nodes (molecule images on their own axes, on top).
    for n in nodes:
        x, y = n["pos"]
        w, h = size[n["id"]]
        ax = fig.add_axes([x - w / 2, y - h / 2, w, h])
        ax.set_xticks([])
        ax.set_yticks([])
        if n.get("png") is not None:
            ax.imshow(plt.imread(io.BytesIO(n["png"]), format="png"), aspect="auto")
        else:
            ax.text(0.5, 0.5, "structure\nunavailable", ha="center", va="center",
                    fontsize=8, color="#9aa3ad", transform=ax.transAxes)
        lw = root_frame_lw if n.get("is_root") else node_frame_lw
        for s in ax.spines.values():
            # Force visible: a caller's global rcParams may disable top/right
            # spines (e.g. the benchmark report style), which would otherwise
            # leave each node framed as an L-bracket instead of a full box.
            s.set_visible(True)
            s.set_edgecolor(n.get("frame_color", "#333333"))
            s.set_linewidth(lw)
        if n.get("caption"):
            bg.text(x, y - h / 2 - 0.014, n["caption"], ha="center", va="top",
                    fontsize=caption_fontsize, color="#20262e", zorder=4,
                    fontweight="bold" if n.get("is_root") else "normal")

    if colorbar is not None:
        cax = fig.add_axes([0.28, 0.045, 0.44, 0.017])
        cb = fig.colorbar(colorbar.mappable(), cax=cax, orientation="horizontal")
        cb.set_label(colorbar_label, fontsize=10, color="#3a4048")
        cb.ax.tick_params(labelsize=10, colors="#3a4048")

    if title:
        tx, tha = (0.02, "left") if title_loc == "left" else (0.5, "center")
        fig.text(tx, 0.975, title, ha=tha, va="top", fontsize=17, fontweight="bold", color="#20262e")
    if subtitle:
        tx, tha = (0.02, "left") if title_loc == "left" else (0.5, "center")
        fig.text(tx, 0.948, subtitle, ha=tha, va="top", fontsize=11.5, color="#5a6470")

    if legend:
        anchor = legend_anchor if legend_anchor is not None else (0.01, 0.005)
        if legend_style == "swatch":
            from matplotlib.patches import Patch
            handles = [Patch(facecolor=c, edgecolor="none", label=lbl)
                       for c, ls, lbl in legend]
        else:
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], color=c, lw=2.4, linestyle=ls, label=lbl)
                       for c, ls, lbl in legend]
        bg.legend(handles=handles, loc=legend_loc, bbox_to_anchor=anchor,
                  frameon=False, fontsize=9, ncol=legend_ncol,
                  handlelength=1.3, columnspacing=1.6, handletextpad=0.5)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        tgt = out_path.with_suffix(f".{fmt}")
        fig.savefig(tgt, dpi=170, facecolor="white")
        written.append(tgt)
    plt.close(fig)
    return written


# --------------------------------------------------------------------------
# One-step radial network
# --------------------------------------------------------------------------
def _onestep_dE(c: dict):
    """Product ΔE vs the root reactant (kcal/mol), or None."""
    v = c.get("product_delta_e")
    return v if isinstance(v, (int, float)) else None


def build_onestep_network(
    summary: dict,
    case_dir: str | Path,
    out_path: str | Path,
    *,
    top: int = 12,
    show_all: bool = False,
    cap: float = 1000.0,
) -> list[Path] | None:
    """Root reactant → one-step products, radial. Returns written paths or None
    when there are no plottable one-step products.

    Default set is the pipeline's validated products (``ranking_included``);
    ``show_all`` loosens to any converged product with a sane ``|ΔE| < cap``.
    """
    case_dir = Path(case_dir)
    case = summary.get("case", case_dir.name)
    formula = summary.get("formula", "")

    depth1 = [c for c in summary.get("candidates", []) if c.get("path_depth") == 1]
    if show_all:
        pool = [c for c in depth1
                if str(c.get("status", "")).startswith("converged")
                and _onestep_dE(c) is not None and abs(_onestep_dE(c)) < cap]
    else:
        pool = [c for c in depth1 if c.get("ranking_included") and _onestep_dE(c) is not None]
    pool = sorted(pool, key=_onestep_dE)
    n_total = len(pool)
    chosen = pool if show_all else pool[:top]
    if not chosen:
        return None

    root = root_structure(summary, case_dir)
    color = DeltaEColor([_onestep_dE(c) for c in chosen])

    nodes = [{
        "id": ROOT,
        "pos": (0.5, 0.52),
        "png": node_png(root, (), size=(470, 400)) if root else None,
        "caption": f"reactant\n{formula}",
        "frame_color": _ROOT_FRAME,
        "is_root": True,
        "w": 0.19,
    }]
    edges = []

    n = len(chosen)
    radius = 0.35
    # node size shrinks if the ring gets crowded, so products never overlap
    arc = 2 * math.pi * radius / max(n, 1)
    node_w = max(0.085, min(0.145, 0.80 * arc))
    root_syms = root.symbols if root is not None else None

    for i, c in enumerate(chosen):
        ang = math.pi / 2 - 2 * math.pi * i / n  # start at top, go clockwise
        x = 0.5 + radius * math.cos(ang)
        y = 0.52 + radius * math.sin(ang)
        reactant, product = candidate_structures(c, case_dir)
        formed, _broken = bond_change(reactant, product) if (reactant and product) else ([], [])
        dE = _onestep_dE(c)
        cid = c.get("candidate_id", f"c{i}")
        nodes.append({
            "id": cid,
            "pos": (x, y),
            "png": node_png(product, formed, size=(430, 360)),
            "caption": f"{cid}\nΔE {dE:+.1f}",
            "frame_color": color.hex(dE),
            "w": node_w,
        })
        rxn = step_label(c, root_syms)
        barrier = c.get("local_ts_barrier_e")
        lbl = rxn
        if isinstance(barrier, (int, float)):
            lbl += f"\n‡{barrier:.0f}"
        edges.append({"src": ROOT, "dst": cid, "label": lbl, "color": "#8a93a0", "width": 1.5})

    kind = "converged" if show_all else "validated (ranking_included)"
    shown = f"all {n}" if show_all else f"top {n}"
    return draw_network(
        nodes, edges, out_path,
        title=f"{case} — one-step reaction network",
        subtitle=f"root reactant → {shown} of {n_total} {kind} one-step products · nodes colored by ΔE, ‡ = SE-GSM barrier",
        figsize=(14, 14), node_w=node_w, colorbar=color,
        colorbar_label="product ΔE vs reactant (kcal/mol)",
    )


# --------------------------------------------------------------------------
# Multi-step layered pathway tree
# --------------------------------------------------------------------------
def _multistep_cum(c: dict):
    v = c.get("path_product_delta_e")
    return v if isinstance(v, (int, float)) else None


def _build_tree(summary: dict, top: int):
    """Return (included, kept_ids, parent_of, children_of) for the pruned tree."""
    included = {c["candidate_id"]: c for c in summary.get("candidates", [])
                if c.get("ranking_included") and _multistep_cum(c) is not None}
    kids = defaultdict(list)
    for cid, c in included.items():
        kids[c.get("parent_id") or ROOT].append(cid)
    for p in kids:
        kids[p].sort(key=lambda cid: _multistep_cum(included[cid]))
        del kids[p][top:]
    # keep only nodes reachable from root through the pruned edges
    kept, parent_of, children_of = [ROOT], {}, defaultdict(list)
    stack = [ROOT]
    while stack:
        nid = stack.pop()
        for ch in kids.get(nid, []):
            parent_of[ch] = nid
            children_of[nid].append(ch)
            kept.append(ch)
            stack.append(ch)
    return included, kept, parent_of, children_of


def _layout(children_of, depth_of, max_depth,
            x_lo=0.10, x_hi=0.8, y_lo=0.13, y_hi=0.90):
    """Tidy layered layout: x by depth, y by DFS leaf order (parents centered).

    The columns sit in ``[x_lo, x_hi]`` and the rows in the band ``[y_lo, y_hi]``;
    both leave a margin so the box half-widths clear the canvas edges, the top
    title/subtitle, and the ΔE colorbar at the bottom.
    """
    yraw, row = {}, [0]

    def dfs(nid):
        ch = children_of.get(nid, [])
        if not ch:
            yraw[nid] = row[0]
            row[0] += 1
            return yraw[nid]
        ys = [dfs(k) for k in ch]
        yraw[nid] = sum(ys) / len(ys)
        return yraw[nid]

    dfs(ROOT)
    n_rows = max(row[0], 1)
    dx = (x_hi - x_lo) / max(max_depth, 1)
    pos = {}
    for nid, yr in yraw.items():
        x = x_lo + depth_of[nid] * dx
        y = y_hi - (yr / max(n_rows - 1, 1)) * (y_hi - y_lo) if n_rows > 1 else 0.5
        pos[nid] = (x, y)
    return pos, n_rows


def build_multistep_network(
    summary: dict,
    case_dir: str | Path,
    out_path: str | Path,
    *,
    top: int = 3,
) -> list[Path] | None:
    """Layered multi-step pathway tree. Returns written paths, or None when the
    search has no multi-step (``ranking_included``, depth ≥ 2) pathways.

    Molecule boxes are sized to dominate the reaction labels, captions are a
    single line (so they never collide with the row below), and the vertical band
    is chosen adaptively so the top box clears the subtitle and the bottom caption
    clears the colorbar for any ``top`` / row count.
    """
    case_dir = Path(case_dir)
    case = summary.get("case", case_dir.name)
    formula = summary.get("formula", "")

    included, kept, parent_of, children_of = _build_tree(summary, top)
    if len(kept) <= 1:
        return None

    depth_of = {ROOT: 0}
    for cid in kept:
        if cid != ROOT:
            depth_of[cid] = included[cid].get("path_depth", 1)
    max_depth = max(depth_of.values())
    if max_depth < 2:  # only one-step products survived — the onestep view covers it
        return None
    n_rows = sum(1 for n in kept if not children_of.get(n))  # leaf rows

    # Figure size: width grows per depth column, height per leaf row (capped so a
    # bushy tree never explodes vertically). Taller rows keep the molecules
    # physically large relative to the fixed-point label text.
    fig_w = 6.5 + 3.8 * max_depth
    fig_h = min(24.0, 1.8 + 2.1 * max(n_rows, 1))

    # Molecule box shape + depiction resolution. node_aspect = box height/width
    # (lower = wider box); render each depiction at that aspect so the structure
    # fills the box without stretching.
    node_aspect, node_px = 360.0 / 430.0, 440
    prod_png = (max(1, round(node_px / node_aspect)), node_px)
    root_png = (max(1, round(node_px * 1.09 / node_aspect)), round(node_px * 1.09))

    # Adaptive vertical band: size node_h so the TOP box lands at y_ceiling (just
    # under the subtitle) and the BOTTOM caption at y_floor (just above the
    # colorbar) for any row count; then place node centers in [c_lo, c_hi].
    y_ceiling, y_floor, caption_gap, caption_fontsize = 0.93, 0.065, 0.014, 10.5
    frac = 0.74  # box height as a fraction of the row pitch
    cap_h = caption_gap + 1.3 * caption_fontsize / 72.0 / fig_h  # 1-line caption height
    if n_rows > 1:
        node_h = min(0.18, frac * (y_ceiling - y_floor - cap_h) / (n_rows - 1 + frac))
    else:
        node_h = 0.18
    c_hi, c_lo = y_ceiling - node_h / 2, y_floor + node_h / 2 + cap_h
    pos, n_rows = _layout(children_of, depth_of, max_depth,
                          x_lo=0.10, x_hi=0.8, y_lo=c_lo, y_hi=c_hi)
    node_w = node_h * fig_h / (fig_w * node_aspect)

    root = root_structure(summary, case_dir)
    color = DeltaEColor([0.0] + [_multistep_cum(included[c]) for c in kept if c != ROOT])

    nodes = [{
        "id": ROOT, "pos": pos[ROOT], "is_root": True, "w": node_w * 1.15,
        "png": node_png(root, (), size=root_png) if root else None,
        "caption": f"reactant · {formula}", "frame_color": _ROOT_FRAME,
    }]
    edges = []
    root_syms = root.symbols if root is not None else None

    for cid in kept:
        if cid == ROOT:
            continue
        c = included[cid]
        reactant, product = candidate_structures(c, case_dir)
        formed, _b = bond_change(reactant, product) if (reactant and product) else ([], [])
        cum = _multistep_cum(c)
        nodes.append({
            "id": cid, "pos": pos[cid], "w": node_w,
            "png": node_png(product, formed, size=prod_png),
            "caption": f"{cid} · ΣΔE {cum:+.0f}", "frame_color": color.hex(cum),
        })
        # edge label: step reaction + stepwise ΔE (+ barrier if any)
        step_syms = reactant.symbols if reactant is not None else root_syms
        rxn = step_label(c, step_syms)
        step_de = c.get("product_delta_e")
        lbl = rxn
        if isinstance(step_de, (int, float)):
            lbl += f"\nΔE {step_de:+.0f}"
        barrier = c.get("local_ts_barrier_e")
        if isinstance(barrier, (int, float)):
            lbl += f"  ‡{barrier:.0f}"
        edges.append({"src": parent_of[cid], "dst": cid, "label": lbl,
                      "color": "#8a93a0", "width": 1.6, "label_t": 0.60})

    return draw_network(
        nodes, edges, out_path,
        title=f"{case} — multi-step reaction pathway tree",
        subtitle=f"root → up to {max_depth} steps · ≤{top} products/parent · nodes colored by cumulative ΔE, ‡ = SE-GSM barrier",
        figsize=(fig_w, fig_h), node_w=node_w, colorbar=color,
        colorbar_label="cumulative ΔE vs root reactant (kcal/mol)",
        edge_label_fontsize=9.0, edge_box_pad=0.28, edge_box_linewidth=0.8,
        caption_fontsize=caption_fontsize, node_aspect=node_aspect,
        node_frame_lw=2.0, root_frame_lw=2.0,
    )


# --------------------------------------------------------------------------
# Composite network: several independent SE-GSM searches, molecule nodes
# --------------------------------------------------------------------------
# Feasible / attempted edge colors (shared with the hand-drawn accident figures).
_FEASIBLE_COLOR = "#2C7FB8"
_ATTEMPTED_COLOR = "#DD8452"
_NEUTRAL_FRAME = "#333333"


def _find_candidate(summary: dict, candidate_id: str | None) -> dict | None:
    """First candidate matching ``candidate_id`` (or the first candidate)."""
    cands = summary.get("candidates", [])
    if candidate_id is None:
        return cands[0] if cands else None
    for c in cands:
        if c.get("candidate_id") == candidate_id:
            return c
    return None


def _species_node(sp: dict, base_dir: Path) -> dict:
    """Build one molecule node from a species spec.

    A species spec has an ``id`` and a ``caption`` plus a structure source:

    - ``xyz``: path to an XYZ (absolute, or relative to ``base_dir``); or
    - ``run_dir`` + optional ``candidate_id`` + ``which`` (``"reactant"`` or
      ``"product"``, default ``"reactant"``): pull the frame from a search run.

    Optional keys: ``pos`` (x, y in [0, 1]), ``w`` (node width fraction),
    ``frame_color`` (hex; ``highlight=True`` is shorthand for the formed-bond
    green), ``highlight_bonds`` (list of 0-based (i, j) to color on the node).
    """
    structure = None
    which = sp.get("which", "reactant")
    if sp.get("xyz"):
        p = Path(sp["xyz"])
        if not p.is_absolute():
            p = base_dir / p
        structure = _first_structure(p)
    elif sp.get("run_dir"):
        rd = Path(sp["run_dir"])
        if not rd.is_absolute():
            rd = base_dir / rd
        try:
            summary, case_dir = load_search(rd)
        except FileNotFoundError:
            summary, case_dir = {}, rd
        cand = _find_candidate(summary, sp.get("candidate_id"))
        if cand is not None:
            reactant, product = candidate_structures(cand, case_dir)
            structure = product if which == "product" else reactant
        if structure is None and which == "reactant":
            structure = root_structure(summary, case_dir)

    highlight = sp.get("highlight_bonds", ()) or ()
    frame_color = sp.get("frame_color")
    if frame_color is None:
        frame_color = _FORMED_COLOR if sp.get("highlight") else _NEUTRAL_FRAME

    return {
        "id": sp["id"],
        "pos": sp.get("pos", (0.5, 0.5)),
        "png": node_png(structure, highlight, size=(470, 400)),
        "caption": sp.get("caption", sp["id"]),
        "frame_color": frame_color,
        "w": sp.get("w", 0.17),
        "is_root": sp.get("is_root", False),
    }


def _edge_energies(ed: dict, base_dir: Path):
    """(barrier, delta_e) for an edge, explicit values overriding a run lookup.

    An edge may name a run via ``run_dir`` (+ optional ``candidate_id``); the
    barrier is that candidate's ``local_ts_barrier_e`` and ΔE its
    ``path_product_delta_e`` (reactant-referenced) unless ``energy_ref`` is
    ``"stepwise"`` (``product_delta_e``). Explicit ``barrier`` / ``delta_e`` on
    the edge win. Returns floats or ``None``.
    """
    barrier = ed.get("barrier")
    delta_e = ed.get("delta_e")
    driving = ed.get("driving_coords")
    if (barrier is None or delta_e is None or ed.get("name") is None) and ed.get("run_dir"):
        rd = Path(ed["run_dir"])
        if not rd.is_absolute():
            rd = base_dir / rd
        try:
            summary, _ = load_search(rd)
        except FileNotFoundError:
            summary = {}
        cand = _find_candidate(summary, ed.get("candidate_id"))
        if cand is not None:
            if barrier is None:
                barrier = cand.get("local_ts_barrier_e")
            if delta_e is None:
                key = "product_delta_e" if ed.get("energy_ref") == "stepwise" else "path_product_delta_e"
                delta_e = cand.get(key)
            if driving is None:
                driving = cand.get("driving_coords")
    barrier = barrier if isinstance(barrier, (int, float)) else None
    delta_e = delta_e if isinstance(delta_e, (int, float)) else None
    return barrier, delta_e, driving


def build_composite_network(
    spec: dict,
    out_path: str | Path,
    *,
    base_dir: str | Path = ".",
    figsize=(12.0, 7.5),
) -> list[Path]:
    """Draw a reaction network across several *independent* SE-GSM searches,
    with molecule structures as nodes and labelled reaction arrows as edges.

    Unlike :func:`build_onestep_network` / :func:`build_multistep_network`
    (one rooted search each), this stitches distinct runs — forward and retro,
    different reactants — into one hand-laid network, e.g. a styrene-dimer map
    where 2 styrene, 1,2-diphenylcyclobutane and the Diels-Alder adduct are all
    separate searches. Structures and energies are read from each run's
    ``candidate_search_summary.json``; nothing is recomputed.

    ``spec`` keys:
      ``species``   list of species node specs (see :func:`_species_node`).
      ``reactions`` list of edge specs: ``src``/``dst`` (species ids), ``name``
                    (reaction-type label), energy source (see
                    :func:`_edge_energies`), ``feasible`` (bool -> default
                    color + solid/dashed), and optional passthrough
                    ``color``/``style``/``rad``/``label_t``/``label_off``/
                    ``note`` (replaces the ΔE lines, e.g. "no clean path").
      ``title`` / ``subtitle`` / ``legend`` (list of (color, ls, label)) /
      ``title_loc`` are forwarded to :func:`draw_network`.

    ``base_dir`` resolves relative ``xyz`` / ``run_dir`` paths in the spec.
    """
    base_dir = Path(base_dir)
    nodes = [_species_node(sp, base_dir) for sp in spec.get("species", [])]

    edges = []
    for ed in spec.get("reactions", []):
        barrier, delta_e, driving = _edge_energies(ed, base_dir)
        name = ed.get("name")
        if name is None and driving:
            name = _describe_reaction(driving)
        lines = [name] if name else []
        if ed.get("note"):
            lines.append(ed["note"])
        else:
            if barrier is not None:
                lines.append(f"ΔE‡ = {barrier:.1f}")
            if delta_e is not None:
                lines.append(f"ΔE = {delta_e:+.1f}".replace("-", "\u2212"))
        feasible = ed.get("feasible", True)
        color = ed.get("color") or (_FEASIBLE_COLOR if feasible else _ATTEMPTED_COLOR)
        style = ed.get("style") or ("-" if feasible else (0, (4, 3)))
        edges.append({
            "src": ed["src"], "dst": ed["dst"], "label": "\n".join(lines),
            "color": color, "style": style, "width": ed.get("width", 2.0),
            "rad": ed.get("rad", 0.0), "label_t": ed.get("label_t", 0.5),
            "label_off": ed.get("label_off", 0.0),
        })

    return draw_network(
        nodes, edges, out_path,
        title=spec.get("title", ""), subtitle=spec.get("subtitle", ""),
        figsize=figsize, legend=spec.get("legend"),
        title_loc=spec.get("title_loc", "center"),
    )
