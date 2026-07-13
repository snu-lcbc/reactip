"""Reporting orchestration for ReactIP sampled searches.

This is the *plotting* half of the pipeline, decoupled from SE-GSM computation.
It consumes only raw outputs already on disk (``candidate_search_summary.json``
plus each candidate's trajectory files) and (re)generates all figures, so it can
run standalone, any number of times, without recomputing.

Both ``scripts/report_sampled_search.py`` (standalone CLI) and the ``--report``
default of ``run_se_gsm.py`` call :func:`generate_sampled_search_report`.

Outputs:
- Per candidate, into that candidate's own run directory:
  ``reaction.png``, ``trajectory.gif``, and ``energy_profile.{png,pdf}``
  as a fallback when no GIF can be produced.
- Per search, into ``<run-dir>/figures/``: ``population.{png,pdf}`` and
  ``reaction_pathway.png`` (+ ``.gif``) for the *successful* carried-forward
  path only (not the full candidate fan-out).
- ``report.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import pathway_2d
from . import visualize as viz
from .sampling import compute_boltzmann_populations
from .utils import (
    MoldenTrajectory,
    TrajectoryFrame,
    parse_molden_xyz_trajectory,
    render_trajectory_gif,
    resolve_trajectory_source,
)

# Default seconds per frame for trajectory GIFs. The original renderer used
# 0.85 s; a touch slower makes bond changes easier to follow.
DEFAULT_GIF_FRAME_SECONDS = 1.0


_STATUS_SHORT = {
    "converged_ts": "converged, TS found",
    "converged_no_ts": "converged, no TS",
    "ran_out_with_ts_candidate": "NOT converged (ran out, TS candidate)",
    "ran_out_no_ts": "NOT converged (ran out)",
    "ended_early_with_ts_candidate": "NOT converged (ended early)",
    "ended_early_no_ts": "NOT converged (ended early)",
    "completed_with_ts_candidate": "completed",
    "completed_no_ts": "completed",
    "runtime_error_after_partial_output": "FAILED (runtime error)",
    "runtime_error_before_candidate_run": "FAILED (setup error)",
}


def _short_status(status) -> str:
    return _STATUS_SHORT.get(str(status), str(status))


def _structure_from_xyz(path):
    frames = viz.read_xyz_frames(path) if path and Path(path).exists() else []
    return frames[0] if frames else None


def _resolve_local_path(path, case_dir: Path) -> Path | None:
    """Resolve a path recorded in a summary after the run tree was copied locally."""
    if not path:
        return None
    p = Path(path)

    parts = p.parts
    for i, part in enumerate(parts):
        if part == case_dir.name or part.startswith(f"{case_dir.name}_"):
            rel = Path(*parts[i + 1 :])
            local = case_dir / rel
            if local.exists():
                return local
    if p.exists():
        return p
    return p


def _candidate_run_dir(candidate: dict, case_dir: Path) -> Path:
    resolved = _resolve_local_path(candidate.get("run_directory"), case_dir)
    return resolved if resolved is not None else Path(candidate.get("run_directory", "."))


def describe_reaction(driving_coords, symbols=None) -> str:
    """Turn raw driving coordinates into a concise, readable reaction label.

    ``ADD 1 3`` -> ``form C1-O3``; ``BREAK 4 5`` -> ``break C4-O5`` when atom
    symbols are available; otherwise falls back to the bare atom indices.
    Multiple coordinates are joined with commas, e.g. ``break C4-O5, form C4-C6``.
    """
    verbs = {"ADD": "form", "BREAK": "break"}
    parts: list[str] = []
    for line in driving_coords or []:
        toks = str(line).split()
        if len(toks) < 3:
            parts.append(str(line))
            continue
        op, a, b = toks[0].upper(), toks[1], toks[2]
        try:
            i, j = int(a), int(b)
        except ValueError:
            parts.append(str(line))
            continue

        def tag(idx: int) -> str:
            if symbols and 1 <= idx <= len(symbols):
                return f"{symbols[idx - 1]}{idx}"
            return str(idx)

        parts.append(f"{verbs.get(op, op.lower())} {tag(i)}-{tag(j)}")
    return ", ".join(parts)


def _candidate_molden(candidate: dict, case_dir: Path) -> MoldenTrajectory | None:
    """Parse a candidate's SE-GSM trajectory into a MoldenTrajectory, or None.

    Resolves the trajectory *locally* via :func:`resolve_trajectory_source`, so a
    stale absolute ``trajectory_source`` recorded on the compute node (e.g. a
    ``/…/cscratch/jobs/<uuid>/…`` path) still finds the copied-over grown-string
    XYZ. This is why per-candidate ``trajectory.gif`` files went missing when the
    report ran against a relocated run tree.
    """
    run_dir = _candidate_run_dir(candidate, case_dir)
    recorded = None
    run_id = 0
    summary_path = _resolve_local_path(candidate.get("summary_path"), case_dir)
    if summary_path and Path(summary_path).exists():
        try:
            summary = json.loads(Path(summary_path).read_text())
        except Exception:
            summary = {}
        recorded = summary.get("trajectory_source")
        run_id = (summary.get("se_gsm_parameters") or {}).get("run_id", 0)
    source = resolve_trajectory_source(run_dir, recorded, run_id)
    if source is None:
        return None
    try:
        return parse_molden_xyz_trajectory(source)
    except Exception:
        return None


def _render_candidate_figures(
    candidate: dict,
    *,
    case_dir: Path,
    make_gif: bool,
    gif_seconds: float,
) -> None:
    """Render per-candidate output into the candidate's own run directory.

    The trajectory GIF (the original 3D-perspective renderer from
    ``reactip.utils``: ball-and-stick + stats panel + synchronized energy curve)
    is the primary artifact, so a separate static energy plot is redundant. The
    static ``energy_profile.png`` is written only as a fallback when no GIF can
    be produced (GIFs disabled, imageio missing, or no trajectory frames).
    """
    run_dir = _candidate_run_dir(candidate, case_dir)
    summary_path = _resolve_local_path(candidate.get("summary_path"), case_dir)
    summary = (
        json.loads(Path(summary_path).read_text())
        if summary_path and Path(summary_path).exists()
        else {}
    )
    coords = "; ".join(candidate.get("driving_coords") or [])
    cid = candidate.get("candidate_id")

    reactant = _structure_from_xyz(_resolve_local_path(candidate.get("reactant_xyz"), case_dir))
    product = _structure_from_xyz(_resolve_local_path(candidate.get("product_xyz"), case_dir))
    if reactant is None:
        reactant = _structure_from_xyz(run_dir / "reactant.xyz")
    if product is None:
        product = _structure_from_xyz(run_dir / "product.xyz")
    reaction_png = run_dir / "reaction.png"
    if reactant is not None and product is not None and not reaction_png.exists():
        structure_symbols = reactant.symbols
        label = describe_reaction(candidate.get("driving_coords"), structure_symbols)
        de = candidate.get("product_delta_e")
        subtitle_parts = [p for p in (label, _short_status(candidate.get("status"))) if p]
        if isinstance(de, (int, float)):
            subtitle_parts.append(f"step dE {de:+.1f} kcal/mol")
        viz.render_reaction_change(
            reactant,
            product,
            reaction_png,
            title=str(cid),
            subtitle=" | ".join(subtitle_parts),
        )

    gif_written = False
    gif_path = run_dir / "trajectory.gif"
    if make_gif and gif_path.exists():
        gif_written = True
    elif make_gif:
        trajectory = _candidate_molden(candidate, case_dir)
        if trajectory is not None and trajectory.frames:
            symbols = trajectory.frames[0].symbols
            label = describe_reaction(candidate.get("driving_coords"), symbols)
            # Carry the SE-GSM outcome in the title: a GIF exists for ANY
            # candidate that produced trajectory frames, including failed or
            # non-converged ones, so the animation must state its own status.
            status = _short_status(candidate.get("status"))
            try:
                render_trajectory_gif(
                    trajectory, gif_path,
                    title=f"{cid}  ({label})  [{status}]", duration=gif_seconds,
                )
                gif_written = True
            except ModuleNotFoundError as exc:
                # Optional plotting deps (imageio) absent in this env — e.g. a
                # GPU compute env that never installed them. Surface it and fall
                # back to the static energy profile; regenerate later with
                # scripts/make_trajectory_gifs.py from an env that has imageio.
                print(f"  [warn] {cid}: GIF skipped ({exc}); energy_profile fallback")
                gif_written = False
            except Exception as exc:  # never abort the whole report on one bad frame
                print(
                    f"  [warn] {cid}: GIF render failed "
                    f"({exc.__class__.__name__}: {exc}); energy_profile fallback"
                )
                gif_written = False

    if not gif_written:
        energies = summary.get("energies") or []
        if energies and not (run_dir / "energy_profile.png").exists():
            viz.plot_energy_profile(
                energies,
                run_dir / "energy_profile",
                title=f"{cid}  [{coords}]",
                ts_node=summary.get("ts_node"),
                reactant_node=summary.get("reactant_node"),
                product_node=summary.get("product_node"),
            )


def _top_products(summary: dict, limit: int = 10, iteration: int | None = None) -> list[dict]:
    """Boltzmann-ranked top products recomputed directly from the candidates.

    Independent of the run's ``print_top`` cap, and works on partial
    (``search_in_progress``) summaries: when no candidate carries a final
    ``ranking_included`` verdict yet, a conservative fallback filter (converged,
    finite score, real bond change, no runtime error) is applied. With
    ``iteration`` set, only that iteration's candidates are ranked and the
    populations are renormalized within the iteration.
    """
    candidates = summary.get("candidates", [])
    if iteration is not None:
        candidates = [c for c in candidates if c.get("iteration") == iteration]
    included = [
        c for c in candidates
        if c.get("ranking_included") and isinstance(c.get("ranking_score_delta_e"), (int, float))
    ]
    if not included:
        included = [
            c for c in candidates
            if isinstance(c.get("ranking_score_delta_e"), (int, float))
            and not c.get("runtime_error")
            and str(c.get("status", "")).startswith("converged")
            and ((c.get("product_bond_added") or 0) + (c.get("product_bond_removed") or 0)) > 0
        ]
    if not included:
        return []
    temperature = (summary.get("search_parameters") or {}).get("temperature") or 298.15
    populations = compute_boltzmann_populations(
        [float(c["ranking_score_delta_e"]) for c in included],
        temperature=float(temperature),
    )
    rows = [
        {**c, "relative_population": p["relative_population"]}
        for c, p in zip(included, populations)
        if p["relative_population"] is not None
    ]
    rows.sort(key=lambda r: (-float(r["relative_population"]), float(r["ranking_score_delta_e"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows[:limit]


def _success_chain(summary: dict) -> list[str]:
    """Candidate IDs of the carried-forward successful (ranked) top-1 path."""
    top = (summary.get("overall_top_candidates") or [None])[0]
    if top and top.get("path_candidate_ids"):
        return list(top["path_candidate_ids"])
    ranked = [c for c in summary.get("candidates", []) if c.get("ranking_included")]
    if not ranked:
        return []
    deepest = max(
        ranked,
        key=lambda c: (int(c.get("path_depth") or 0), -float(c.get("ranking_score_delta_e") or 0.0)),
    )
    return list(deepest.get("path_candidate_ids") or [deepest.get("candidate_id")])


def _render_pathway(summary: dict, figures_dir: Path, *, make_gif: bool, gif_seconds: float) -> list[str]:
    by_id = {str(c.get("candidate_id")): c for c in summary.get("candidates", [])}
    chain = [cid for cid in _success_chain(summary) if cid in by_id]
    if not chain:
        return ["No successful (ranked) multi-step path to render."]

    case_dir = figures_dir.parent
    # 2D chemical-structure reaction graph (reactant -> (TS) -> product drawn as
    # skeletal structures with formed/broken bonds highlighted and CIP stereo —
    # R/S centres, E/Z double bonds — annotated). This is the static-structure
    # depiction of the carried-forward pathway (the old 3D ball-and-stick
    # reaction_pathway.png was removed in favour of the reaction_network fan-out
    # figures). RDKit two-tier perception degrades to a connectivity graph on
    # distorted geometries, so this is wrapped defensively.
    try:
        p2d_nodes: list[pathway_2d.PathNode] = [
            pathway_2d.PathNode(
                xyz_path=str(summary.get("initial_xyz_file")),
                label=f"reactant ({summary.get('formula','')})",
                delta_e=0.0,
                charge=int(summary.get("search_parameters", {}).get("charge", 0) or 0),
            )
        ]
        p2d_steps: list[pathway_2d.PathStep] = []
        charge = int(summary.get("search_parameters", {}).get("charge", 0) or 0)
        for cid in chain:
            c = by_id[cid]
            driving = c.get("driving_coords") or []
            barrier = c.get("local_ts_barrier_e")
            # optional TS node for this step
            ts_path = None
            summary_path = _resolve_local_path(c.get("summary_path"), case_dir)
            if summary_path and Path(summary_path).exists():
                cand_summary = json.loads(Path(summary_path).read_text())
                ts_path = _resolve_local_path(
                    (cand_summary.get("raw_output_paths") or {}).get("ts_node_xyz"),
                    case_dir,
                )
            if ts_path and Path(ts_path).exists():
                parent_de = c.get("parent_path_product_delta_e") or 0.0
                ts_de = (float(parent_de) + float(barrier)
                         if isinstance(barrier, (int, float)) else None)
                p2d_nodes.append(pathway_2d.PathNode(
                    xyz_path=str(ts_path), label=f"TS ({cid})",
                    delta_e=ts_de, is_ts=True, charge=charge))
                p2d_steps.append(pathway_2d.PathStep(driving_coords=list(driving)))
            product_xyz = _resolve_local_path(c.get("product_xyz"), case_dir)
            if product_xyz:
                is_last = cid == chain[-1]
                p2d_nodes.append(pathway_2d.PathNode(
                    xyz_path=str(product_xyz),
                    label=(f"product ({cid})" if is_last else f"after {cid}"),
                    delta_e=(c.get("path_product_delta_e_abs")
                             if c.get("path_product_delta_e_abs") is not None
                             else c.get("path_product_delta_e")),
                    charge=charge))
                p2d_steps.append(pathway_2d.PathStep(
                    driving_coords=list(driving),
                    barrier_e=(float(barrier) if isinstance(barrier, (int, float)) else None)))
        if len(p2d_nodes) >= 2:
            info = pathway_2d.render_pathway(
                p2d_nodes, p2d_steps, figures_dir / "reaction_pathway_2d.png",
                title=f"{summary.get('case','')}: 2D reaction pathway (top-1 carried forward)",
            )
            graph_flag = any(m == "graph" for m in info.get("perception_modes", []))
    except Exception as exc:  # never let figure generation break the report
        graph_flag = None
        _pathway_2d_error = str(exc)

    # Reaction-coordinate energy diagram for the same chain.
    diagram_steps = []
    for cid in chain:
        c = by_id[cid]
        parent_de = c.get("parent_path_product_delta_e")
        barrier = c.get("local_ts_barrier_e") if c.get("local_ts_barrier_e") is not None else c.get("ts_energy")
        ts_level = None
        if isinstance(parent_de, (int, float)) and isinstance(barrier, (int, float)):
            ts_level = float(parent_de) + float(barrier)
        structure = _structure_from_xyz(_resolve_local_path(c.get("product_xyz"), case_dir))
        symbols = structure.symbols if structure is not None else None
        diagram_steps.append({
            "label": describe_reaction(c.get("driving_coords"), symbols) or cid,
            "product_level": c.get("path_product_delta_e"),
            "ts_level": ts_level,
        })
    if any(s.get("product_level") is not None for s in diagram_steps):
        viz.plot_reaction_path_diagram(
            diagram_steps, figures_dir / "reaction_path_energy",
            title=f"{summary.get('case','')}: multi-step reaction path",
        )

    notes = [f"Successful pathway: root -> {' -> '.join(chain)}"]
    if 'graph_flag' in dir() and graph_flag is not None:
        suffix = (" (some nodes shown as connectivity graphs; distorted "
                  "geometry defeated full bond-order perception)" if graph_flag else "")
        notes.append(f"2D reaction-structure graph rendered{suffix}.")
    if make_gif:
        # Each candidate's frame energies are relative to its own local
        # reactant; shift every segment by the parent's cumulative dE so the
        # combined animation shows one continuous energy profile vs the root
        # reactant (matching the reaction_path_energy diagram).
        combined_frames: list[TrajectoryFrame] = []
        for cid in chain:
            candidate = by_id[cid]
            traj = _candidate_molden(candidate, case_dir)
            if traj is None:
                continue
            offset = candidate.get("parent_path_product_delta_e") or 0.0
            for frame in traj.frames:
                energy = frame.energy_kcal_mol
                combined_frames.append(
                    TrajectoryFrame(
                        symbols=frame.symbols,
                        coordinates=frame.coordinates,
                        energy_kcal_mol=(energy + offset) if energy is not None else None,
                    )
                )
        if combined_frames:
            try:
                render_trajectory_gif(
                    MoldenTrajectory(frames=tuple(combined_frames), metrics={}),
                    figures_dir / "reaction_pathway.gif",
                    title=f"{summary.get('case','')}: full pathway",
                    duration=gif_seconds,
                )
                notes.append("Combined pathway GIF rendered.")
            except ModuleNotFoundError:
                pass
    return notes


def _render_networks(summary: dict, figures_dir: Path) -> list[str]:
    """Render the reaction-network fan-out figures from the search summary.

    - ``network_onestep.png``: the root reactant surrounded by its validated
      one-step products (attempted whenever ≥1 such product exists).
    - ``network_multistep.png``: the layered multi-step pathway tree, rendered
      only when the search actually reached depth ≥ 2.

    These replace the removed 3D ball-and-stick ``reaction_pathway.png`` and use
    the same 2D structure depictions as the rest of the report. Failures are
    recorded as notes and never abort report generation.
    """
    from . import reaction_network as rnet

    case_dir = figures_dir.parent
    notes: list[str] = []
    try:
        written = rnet.build_onestep_network(summary, case_dir, figures_dir / "network_onestep.png")
        notes.append(
            "One-step reaction network rendered (network_onestep.png)."
            if written
            else "No validated one-step products to draw a one-step network."
        )
    except Exception as exc:  # never let a figure abort the report
        notes.append(f"One-step network failed: {exc.__class__.__name__}: {exc}")
    try:
        written = rnet.build_multistep_network(summary, case_dir, figures_dir / "network_multistep.png")
        if written:
            notes.append("Multi-step reaction pathway tree rendered (network_multistep.png).")
    except Exception as exc:
        notes.append(f"Multi-step network failed: {exc.__class__.__name__}: {exc}")
    return notes


def _write_report_md(
    summary: dict,
    run_dir: Path,
    pathway_notes: list[str],
    top_products: list[dict],
    root_symbols=None,
) -> Path:
    params = summary.get("search_parameters") or {}
    lines = [
        f"# Sampled search report: {summary.get('case','')}",
        "",
        f"- Formula: `{summary.get('formula','')}`",
        f"- Candidates: {summary.get('candidate_count')} | ranked: {summary.get('ranked_candidate_count')}",
        f"- Multi-step depth reached: {summary.get('max_path_depth', '?')} "
        f"(requested {summary.get('iterations_requested', '?')} iterations)"
        + (f" — **truncated early**: {summary.get('stop_reason')}" if summary.get("stopped_early") else ""),
        f"- Score mode: {params.get('sample_score_mode')}"
        + (" | exhaustive first iteration" if params.get("sample_all_first_iteration") else "")
        + (" | exhaustive all iterations" if params.get("sample_all_driving_coords") else ""),
        f"- OOD clamp: {params.get('sample_max_abs_delta_e')} kcal/mol; "
        f"dedupe: {params.get('sample_dedupe_products')}; TS verify: {params.get('sample_verify_ts')}",
    ]
    if summary.get("search_in_progress"):
        lines.append("- **Note: partial summary — the search was still in progress when this was written.**")
    def _product_table(rows: list[dict]) -> list[str]:
        out = [
            "| rank | candidate | path | reaction | dE (kcal/mol) | population | status | TS imag modes |",
            "|---:|---|---|---|---:|---:|---|---:|",
        ]
        for t in rows:
            pop = t.get("relative_population")
            score = t.get("ranking_score_delta_e")
            im = t.get("ts_imaginary_mode_count")
            out.append(
                "| {r} | {c} | {p} | {rx} | {de} | {pop} | {s} | {im} |".format(
                    r=t.get("rank"), c=t.get("candidate_id"),
                    p="->".join(t.get("path_candidate_ids") or []),
                    rx=describe_reaction(t.get("driving_coords"), root_symbols),
                    de=f"{score:.1f}" if isinstance(score, (int, float)) else "n/a",
                    pop=f"{100*pop:.1f}%" if isinstance(pop, (int, float)) else "n/a",
                    s=_short_status(t.get("status")), im=im if im is not None else "-",
                )
            )
        if len(out) == 2:
            out.append("| - | (no ranked products) | | | | | | |")
        return out

    lines += [
        "",
        "## Top products — global Boltzmann ranking (all iterations, dE vs root reactant)",
        "",
    ]
    lines += _product_table(top_products)

    iterations_present = sorted({
        int(c.get("iteration") or 0) for c in summary.get("candidates", [])
    })
    for it in iterations_present:
        rows = _top_products(summary, limit=10, iteration=it)
        n_total = sum(1 for c in summary.get("candidates", []) if c.get("iteration") == it)
        lines += [
            "",
            f"## Iteration {it} Boltzmann ranking "
            f"({len(rows)} ranked of {n_total} candidates; populations normalized within this iteration)",
            "",
        ]
        lines += _product_table(rows)
    lines += ["", "## Pathway", ""] + [f"- {n}" for n in pathway_notes]
    lines += [
        "", "## Figures", "",
        "- `figures/population.png` / `.pdf` — global Boltzmann population distribution (top 10)",
        "- `figures/network_onestep.png` — one-step reaction network: the root "
        "reactant surrounded by its validated one-step products (2D structures, "
        "formed bonds green), edges labelled with the reaction and SE-GSM barrier "
        "(‡), nodes colored by product ΔE",
        "- `figures/network_multistep.png` — layered multi-step reaction pathway "
        "tree, present only when the search reached depth ≥ 2; nodes colored by "
        "cumulative ΔE vs the root reactant",
        "- `figures/reaction_pathway_2d.png` — 2D chemical-structure reaction graph "
        "of the top-1 carried-forward path (skeletal structures with formed bonds "
        "in green / broken bonds in red, and CIP stereochemistry — R/S centres, "
        "E/Z double bonds — annotated; a `[graph]` tag marks nodes drawn as "
        "connectivity graphs where distorted geometry defeated full bond-order "
        "perception)",
        "- `figures/reaction_pathway.gif` — 3D animation of the full carried-forward "
        "path (concatenated, energies referenced to the root reactant)",
        "- `figures/reaction_path_energy.png` / `.pdf` — reaction-coordinate energy diagram",
        "- per candidate, in its own run dir: `reaction.png` (static reactant "
        "-> product bond-change depiction); `trajectory.gif` (3D structure + "
        "synchronized energy profile); `energy_profile.png` only as a fallback "
        "when no GIF could be produced",
        "",
        "## Conventions",
        "",
        "- A `trajectory.gif` exists for **any** candidate whose SE-GSM run saved "
        "trajectory frames — including failed and non-converged ones. GIF presence "
        "does **not** imply success; the SE-GSM outcome is shown in the GIF title "
        "and in the tables above. A candidate with no GIF produced no usable "
        "trajectory at all (hard failure before the string was saved).",
        "- Populations are Boltzmann weights over ranked candidates only "
        "(converged, real bond change, |dE| within the out-of-domain clamp, "
        "de-duplicated by product).",
    ]
    report = run_dir / "report.md"
    report.write_text("\n".join(lines) + "\n")
    return report


def generate_sampled_search_report(
    summary_path: str | Path,
    *,
    make_gif: bool = True,
    candidate_figures: bool = True,
    gif_seconds: float = DEFAULT_GIF_FRAME_SECONDS,
    progress: bool = False,
) -> dict:
    """Generate all figures + report for one sampled search from raw outputs."""
    summary_path = Path(summary_path)
    if summary_path.is_dir():
        summary_path = summary_path / "candidate_search_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No candidate_search_summary.json at {summary_path}")

    summary = json.loads(summary_path.read_text())
    run_dir = summary_path.parent
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # The 3D ball-and-stick reaction_pathway.png was replaced by the reaction
    # network fan-out figures (network_onestep/network_multistep). Remove a stale
    # one left by an older run so regenerated reports don't keep a figure the
    # pipeline no longer produces.
    stale = figures_dir / "reaction_pathway.png"
    if stale.exists():
        stale.unlink()

    if candidate_figures:
        candidates = summary.get("candidates", [])
        for i, candidate in enumerate(candidates, start=1):
            try:
                _render_candidate_figures(
                    candidate,
                    case_dir=run_dir,
                    make_gif=make_gif,
                    gif_seconds=gif_seconds,
                )
            except Exception as exc:
                print(f"  [warn] {candidate.get('candidate_id')}: {exc}")
            if progress:
                print(f"  candidate figures {i}/{len(candidates)}", end="\r")
        if progress:
            print()

    top_products = _top_products(summary, limit=10)
    score_label = (
        "rate-limiting barrier (kcal/mol)"
        if (summary.get("search_parameters") or {}).get("sample_score_mode") == "kinetic"
        else "cumulative dE vs root reactant (kcal/mol)"
    )
    root = _structure_from_xyz(summary.get("initial_xyz_file"))
    root_symbols = root.symbols if root is not None else None
    if top_products:
        for r in top_products:
            r["reaction_label"] = describe_reaction(r.get("driving_coords"), root_symbols)
        viz.plot_population_distribution(
            top_products, figures_dir / "population",
            title=f"{summary.get('case','')}: relative product populations (top {len(top_products)})",
            score_label=score_label,
        )
    pathway_notes = _render_pathway(summary, figures_dir, make_gif=make_gif, gif_seconds=gif_seconds)
    pathway_notes += _render_networks(summary, figures_dir)
    report = _write_report_md(summary, run_dir, pathway_notes, top_products, root_symbols)
    return {
        "report": report,
        "figures_dir": figures_dir,
        "pathway_notes": pathway_notes,
        "top_products": top_products,
    }
