"""2D reaction-pathway depiction for ReactIP.

This module draws a **2D chemical-structure reaction graph** — reactant ->
(TS) -> intermediate -> product as 2D skeletal structures connected by
labelled arrows, the way a mechanism is drawn in a paper, with CIP
stereochemistry (R/S centres, E/Z double bonds) annotated where the geometry
supports it. It is the static-structure complement to the animated
``trajectory.gif`` and the ``reaction_network`` fan-out figures.

Design decisions that make it robust on SE-GSM output:

* **Two-tier structure perception.** SE-GSM product/TS geometries are often
  distorted or open-shell, so full bond-order perception (RDKit
  ``DetermineBonds``) frequently fails -- exactly the limitation
  ``reactip/smiles.py`` already flags. We try full perception first and fall
  back to *connectivity-only* (``DetermineConnectivity``, single lines, no
  orders). A node is ALWAYS drawable; the caption records which mode was used
  so a graph-mode structure is never mistaken for a validated Lewis structure.

* **Bond-change highlighting.** The atoms/bonds named in each step's driving
  coordinates (``ADD i j`` / ``BREAK i j``) are highlighted on the flanking
  structures -- formed bonds green, broken bonds red -- so the depiction shows
  *what the reaction does*, not just the endpoints.

* **Energetics on the arrows.** Each arrow is annotated with the step driving
  coordinates and, when a TS was located, the barrier dE-double-dagger; a TS
  node is drawn with a distinct dashed frame.

* **Pure matplotlib composition** over per-molecule RDKit PNGs -- no graphviz
  or other new dependency (rdkit + matplotlib + numpy + pillow are already
  ReactIP deps); wraps to multiple rows for long paths.

RDKit / matplotlib / PIL imports are all lazy (inside functions) so importing
this module never fails, and a perception failure degrades to a placeholder
box rather than crashing the report. ``render_pathway`` is the low-level API;
``build_pathway_from_summary`` renders any ranked candidate's full root->product
path straight from a ``candidate_search_summary.json``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Structure perception
# --------------------------------------------------------------------------- #
def _plain_xyz_block(text: str) -> str | None:
    """Coerce a (possibly extended-XYZ) file into a plain 4-column XYZ block.

    Trajectory geometries written by the pipeline are *extended* XYZ: each atom
    line carries the element plus x/y/z and then per-atom energy/forces (an
    ASE ``Properties=...`` comment on line 2), e.g.::

        O   2.680 -0.118 -0.049  -435.488  0.013 -0.001 -0.000

    RDKit's ``MolFromXYZBlock`` only accepts plain ``element x y z`` lines and
    returns ``None`` on the extra columns, which is what surfaced as
    "structure unavailable / parse_fail" for the reactant panels. Here we keep
    only the element symbol and the first three float columns per atom.
    """
    lines = text.splitlines()
    if not lines:
        return None
    try:
        n = int(lines[0].split()[0])
    except (ValueError, IndexError):
        return None
    atom_lines = lines[2:2 + n]
    if len(atom_lines) < n:
        return None
    clean = [str(n), "reactip"]
    for ln in atom_lines:
        parts = ln.split()
        if len(parts) < 4:
            return None
        clean.append(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}")
    return "\n".join(clean) + "\n"


def _perceive(xyz_path: str | Path, charge: int = 0):
    """(mol, mode): try full bond orders, fall back to connectivity-only graph.

    A full perception is only accepted when it comes out clean (no forced
    charges/radicals); a charge/radical-separated fit on a distorted geometry is
    treated as low-confidence and downgraded to the connectivity-only skeleton.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    from reactip.visualize import _clean_perception

    p = Path(xyz_path)
    if not p.exists():
        return None, "missing"
    block = _plain_xyz_block(p.read_text())
    if block is None:
        return None, "parse_fail"
    mol = Chem.MolFromXYZBlock(block)
    if mol is None:
        return None, "parse_fail"
    m_full = None
    try:
        m_full = Chem.Mol(mol)
        rdDetermineBonds.DetermineBonds(m_full, charge=charge)
    except Exception:
        m_full = None
    if m_full is not None and _clean_perception(m_full):
        # Perceive stereochemistry (chiral centres + double-bond cis/trans) from
        # the real 3D geometry while the conformer is still attached, so the 2D
        # depiction can annotate R/S and E/Z. Best-effort: distorted SE-GSM /
        # TS geometries may not yield clean stereo, which is acceptable.
        try:
            Chem.AssignStereochemistryFrom3D(m_full)
        except Exception:
            pass
        return m_full, "full"
    try:
        m = Chem.Mol(mol)
        rdDetermineBonds.DetermineConnectivity(m)
        return m, "graph"
    except Exception as exc:  # pragma: no cover
        if m_full is not None:  # last resort: the charge-separated full perception
            return m_full, "full"
        return None, f"err:{type(exc).__name__}"


def _read_xyz_symbols(xyz_path: str | Path) -> list[str]:
    """Element symbols in file order (1-based indexing matches driving coords)."""
    lines = Path(xyz_path).read_text().splitlines()
    n = int(lines[0].split()[0])
    return [ln.split()[0] for ln in lines[2:2 + n]]


def _changed_atoms_bonds(driving_coords: Sequence[str]):
    """Parse ['ADD 1 3', 'BREAK 2 4'] -> ({formed pairs}, {broken pairs}) 0-based."""
    formed, broken = [], []
    for dc in driving_coords or []:
        parts = dc.split()
        if len(parts) != 3:
            continue
        op, i, j = parts[0].upper(), int(parts[1]) - 1, int(parts[2]) - 1
        (formed if op in ("ADD", "FORM") else broken).append((i, j))
    return formed, broken


def _mol_to_image(mol, mode, highlight_formed, highlight_broken, size=(320, 260)):
    """Render a single 2D structure to an RGBA numpy array with bond highlights.

    C–H hydrogens are condensed to implicit for a clean skeletal depiction
    (heteroatom H and any H on a highlighted bond are kept) and the layout is
    done with CoordGen; highlight indices are remapped onto the condensed mol.
    """
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from PIL import Image

    from reactip.visualize import _condense_hs, _layout_2d, _prep_graph_skeleton

    # Keep H atoms that sit on a highlighted formed/broken bond; condense the rest.
    all_pairs = list(highlight_formed) + list(highlight_broken)
    keep_h = {int(a) for pair in all_pairs for a in pair
              if 0 <= int(a) < mol.GetNumAtoms()
              and mol.GetAtomWithIdx(int(a)).GetAtomicNum() == 1}
    m, old2new = _condense_hs(mol, keep_h)

    def _remap(pairs):
        return [(old2new[i], old2new[j]) for (i, j) in pairs
                if i in old2new and j in old2new]

    highlight_formed = _remap(highlight_formed)
    highlight_broken = _remap(highlight_broken)

    # Graph-mode: clean single-bond skeleton (no orders/charges/radical dots).
    if mode == "graph":
        _prep_graph_skeleton(m)
    _layout_2d(m)

    hi_atoms, hi_bonds, bond_colors, atom_colors = [], [], {}, {}
    GREEN, RED = (0.13, 0.62, 0.34), (0.75, 0.22, 0.17)

    def _mark(pairs, color):
        for i, j in pairs:
            if i < m.GetNumAtoms() and j < m.GetNumAtoms():
                hi_atoms.extend([i, j])
                atom_colors[i] = color
                atom_colors[j] = color
                b = m.GetBondBetweenAtoms(i, j)
                if b is not None:
                    hi_bonds.append(b.GetIdx())
                    bond_colors[b.GetIdx()] = color

    _mark(highlight_formed, GREEN)
    _mark(highlight_broken, RED)

    if mode != "graph":
        # full perception: annotate CIP stereo — (R)/(S) chiral centres and
        # (E)/(Z) cis-trans double bonds — using rdCIPLabeler for true CIP
        # priority (the stereo tags were set from 3D in _perceive). Wedge/hash
        # bonds are drawn by PrepareAndDrawMolecule from the chiral tags.
        try:
            from rdkit.Chem import rdCIPLabeler

            rdCIPLabeler.AssignCIPLabels(m)
        except Exception:
            pass

    hi_a, hi_b = hi_atoms or None, hi_bonds or None
    a_col, b_col = atom_colors or None, bond_colors or None

    def _drawer():
        dd = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        o = dd.drawOptions()
        o.bondLineWidth = 2
        o.addStereoAnnotation = mode != "graph"
        return dd

    def _draw_no_prepare():
        m.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(m)
        dd = _drawer()
        dd.drawOptions().prepareMolsBeforeDrawing = False
        try:
            dd.DrawMolecule(
                m, highlightAtoms=hi_a, highlightBonds=hi_b,
                highlightAtomColors=a_col, highlightBondColors=b_col,
            )
        except Exception:
            dd = _drawer()
            dd.drawOptions().prepareMolsBeforeDrawing = False
            dd.DrawMolecule(m)
        return dd

    # Full (clean) perception: prepare (kekulize/wedge) + draw for nice double
    # bonds / aromatics. Graph-mode skeletons skip prepare so sanitize can't
    # re-derive the radicals/charges _prep_graph_skeleton stripped.
    if mode != "graph":
        d = _drawer()
        try:
            rdMolDraw2D.PrepareAndDrawMolecule(
                d, m, highlightAtoms=hi_a, highlightBonds=hi_b,
                highlightAtomColors=a_col, highlightBondColors=b_col,
            )
        except Exception:
            d = _draw_no_prepare()
    else:
        d = _draw_no_prepare()
    d.FinishDrawing()
    png = d.GetDrawingText()
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


# --------------------------------------------------------------------------- #
# Pathway model
# --------------------------------------------------------------------------- #
@dataclass
class PathNode:
    xyz_path: str
    label: str = ""
    delta_e: float | None = None            # kcal/mol vs root
    is_ts: bool = False
    charge: int = 0


@dataclass
class PathStep:
    driving_coords: list[str] = field(default_factory=list)
    barrier_e: float | None = None          # kcal/mol, if a TS was located
    step_delta_e: float | None = None        # kcal/mol for this elementary step


def _pretty_dc(driving_coords, symbols=None):
    """'ADD 1 3' -> 'form C1-O3'; 'BREAK 2 4' -> 'break C2-C4' (symbols optional)."""
    def tag(idx):
        if symbols and 1 <= idx <= len(symbols):
            return f"{symbols[idx - 1]}{idx}"
        return str(idx)
    out = []
    for dc in driving_coords or []:
        p = dc.split()
        if len(p) == 3:
            verb = "form" if p[0].upper() in ("ADD", "FORM") else "break"
            out.append(f"{verb} {tag(int(p[1]))}\u2013{tag(int(p[2]))}")
        else:
            out.append(dc)
    return out


def render_pathway(
    nodes: Sequence[PathNode],
    steps: Sequence[PathStep],
    out_path: str | Path,
    *,
    title: str = "",
    per_row: int = 4,
    node_size=(320, 260),
    dpi: int = 190,
):
    """Compose a 2D reaction-pathway figure from perceived structures.

    ``nodes`` are the stable species (reactant, intermediates, product) plus any
    TS nodes; ``steps[k]`` describes the transformation from ``nodes[k]`` to
    ``nodes[k+1]`` (``len(steps) == len(nodes) - 1``). Returns a dict with the
    per-node perception modes and the output path.
    """
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    try:
        _sym = _read_xyz_symbols(nodes[0].xyz_path)
    except Exception:
        _sym = None

    imgs, modes = [], []
    for nd in nodes:
        mol, mode = _perceive(nd.xyz_path, charge=nd.charge)
        modes.append(mode)
        imgs.append((mol, mode) if mol is not None else None)

    n = len(nodes)
    rows = int(np.ceil(n / per_row))
    ncol = min(per_row, n)
    col_w = 1.4  # horizontal cell pitch (>1 opens a clear gap between molecules for the arrow)
    fig_w = ncol * 3.3 * col_w
    fig_h = rows * 3.15 + 0.7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, ncol * col_w); ax.set_ylim(0, rows); ax.axis("off")

    GREEN, RED = "#218f56", "#bf382c"
    for idx, nd in enumerate(nodes):
        row = idx // per_row
        col = idx % per_row
        # serpentine layout so arrows always point to the physical next cell
        if row % 2 == 1:
            col = (min(per_row, n) - 1) - col if row < rows - 1 or n % per_row == 0 else (n - 1 - idx) % per_row
        x = col * col_w + col_w / 2
        y = rows - 1 - row + 0.5

        # highlight bonds for the leaving step (reactant side) and arriving step
        formed_here, broken_here = [], []
        if idx < len(steps):
            f, b = _changed_atoms_bonds(steps[idx].driving_coords)
            formed_here += f; broken_here += b
        if idx > 0:
            f, b = _changed_atoms_bonds(steps[idx - 1].driving_coords)
            formed_here += f; broken_here += b

        entry = imgs[idx]
        if entry is None:
            ax.add_patch(plt.Rectangle((x - 0.42, y - 0.32), 0.84, 0.64, fill=False,
                                       ec="#bbb", ls="--"))
            ax.text(x, y, "structure\nunavailable", ha="center", va="center",
                    fontsize=7, color="#999")
        else:
            mol, mode = entry
            arr = _mol_to_image(mol, mode, formed_here, broken_here, size=node_size)
            oi = OffsetImage(arr, zoom=0.62)
            ab = AnnotationBbox(
                oi, (x, y + 0.06), frameon=nd.is_ts,
                bboxprops=dict(edgecolor="#e0952a", lw=2.0, ls=(0, (4, 2))) if nd.is_ts else None,
            )
            ax.add_artist(ab)

        cap = nd.label or ("reactant" if idx == 0 else f"node {idx}")
        if nd.is_ts:
            cap = f"\u2021 {cap}"
        de = f"\n\u0394E = {nd.delta_e:+.1f}" if nd.delta_e is not None else ""
        mflag = "  [graph]" if modes[idx] == "graph" else ("" if modes[idx] == "full" else f"  [{modes[idx]}]")
        ax.text(x, y - 0.40, f"{cap}{de}{mflag}", ha="center", va="top", fontsize=7.6)

        if idx < n - 1:
            nrow = (idx + 1) // per_row
            if nrow == row:  # same row -> horizontal arrow, centred in the cell gap
                direction = 1 if (row % 2 == 0) else -1
                mid = x + direction * col_w / 2
                ax.annotate("", xy=(mid + direction * 0.16, y + 0.06),
                            xytext=(mid - direction * 0.16, y + 0.06),
                            arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#333"))
                stp = steps[idx]
                lab = "\n".join(_pretty_dc(stp.driving_coords, _sym))
                if stp.barrier_e is not None:
                    lab = (lab + "\n" if lab else "") + f"\u0394E$^\u2021$={stp.barrier_e:.1f}"
                ax.text(mid, y + 0.24, lab, ha="center", va="bottom",
                        fontsize=6.6, color="#444")
            else:  # wrap: down arrow
                ax.annotate("", xy=(x, y - 0.5), xytext=(x, y - 0.34),
                            arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#333"))

    ax.plot([], [], "s", color=GREEN, label="bond formed (ADD)")
    ax.plot([], [], "s", color=RED, label="bond broken (BREAK)")
    ax.plot([], [], "s", color="#e0952a", label="transition state (\u2021)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"nodes": n, "perception_modes": modes, "out_path": str(out_path)}


def _kc(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_pathway_from_summary(summary_path, candidate_id, out_path, *, title=None):
    """Render the full root->product path of one ranked candidate.

    Reconstructs the node chain from ``path_candidate_ids`` and locates each
    step's reactant/product geometry and driving coordinates from the per-run
    summary records.
    """
    import json

    S = json.loads(Path(summary_path).read_text())
    by = {c["candidate_id"]: c for c in S["candidates"]}
    if candidate_id not in by:
        raise KeyError(candidate_id)
    chain = by[candidate_id].get("path_candidate_ids") or [candidate_id]

    nodes, steps = [], []
    first = by[chain[0]]
    nodes.append(PathNode(first["reactant_xyz"], label="reactant", delta_e=0.0))
    for k, cid in enumerate(chain):
        c = by[cid]
        has_ts = c.get("has_ts") and c.get("ts_node_xyz")
        if has_ts:
            nodes.append(PathNode(c["ts_node_xyz"], label=f"TS ({cid})",
                                  delta_e=_kc(c.get("local_ts_barrier_e")), is_ts=True))
            steps.append(PathStep(driving_coords=c.get("driving_coords") or []))
        nodes.append(PathNode(
            c["product_xyz"],
            label=cid if k < len(chain) - 1 else f"product ({cid})",
            delta_e=_kc(c.get("path_product_delta_e_abs") or c.get("product_delta_e")),
        ))
        steps.append(PathStep(driving_coords=c.get("driving_coords") or [],
                              barrier_e=_kc(c.get("local_ts_barrier_e")),
                              step_delta_e=_kc(c.get("delta_e"))))

    if title is None:
        title = f"{S.get('case', 'reaction')} \u2014 pathway to {candidate_id}"
    return render_pathway(nodes, steps, out_path, title=title)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="2D reaction-pathway depiction")
    ap.add_argument("summary", help="candidate_search_summary.json")
    ap.add_argument("candidate", help="candidate id, e.g. cand_0140")
    ap.add_argument("-o", "--out", default="reaction_pathway_2d.png")
    a = ap.parse_args()
    print(build_pathway_from_summary(a.summary, a.candidate, a.out))
