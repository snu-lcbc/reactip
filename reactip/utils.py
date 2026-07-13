"""Trajectory parsing and export helpers for ReactIP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Covalent radii and the bond-detection cutoff scale are defined ONCE in
# reactip.sampling and imported here so the sampler, the trajectory tools and
# the figures all perceive identical connectivity (see BOND_SCALE).
from reactip.sampling import (  # noqa: E402
    BOND_SCALE,
    COVALENT_RADII as _COVALENT_RADII,
    DEFAULT_COVALENT_RADIUS as _DEFAULT_RADIUS,
)

# CPK-style atom colors for the shaded-sphere ball-and-stick renderer.
_ELEMENT_COLORS = {
    "H": "#f2f2f2",
    "C": "#262626",
    "N": "#2f63d6",
    "O": "#e6402c",
    "F": "#33cc55",
    "S": "#e0c020",
    "Cl": "#28b828",
    "Br": "#a0522d",
}

# Reference bond lengths (Angstrom): (single, double, triple|None); keys are sorted
# element pairs. A measured length below the single/double midpoint is drawn as a
# double bond, below the double/triple midpoint as a triple. Pairs not listed (and
# any bond involving H) are always single. This length heuristic is robust on the
# fragment / radical / transition-state geometries along a reaction string, where
# full valence perception (e.g. RDKit) tends to fail.
_BOND_REF = {
    ("C", "C"): (1.54, 1.34, 1.20),
    ("C", "N"): (1.47, 1.28, 1.16),
    ("C", "O"): (1.43, 1.21, 1.13),
    ("N", "N"): (1.45, 1.25, 1.10),
    ("N", "O"): (1.41, 1.21, None),
    ("O", "O"): (1.48, 1.21, None),
    ("C", "S"): (1.82, 1.60, None),
}

# Ball-and-stick rendering parameters.
_DEFAULT_COLOR = "#9a9a9a"
# _DEFAULT_RADIUS is imported from reactip.sampling (single source of truth).
_ATOM_SCALE = 0.45            # sphere radius as a fraction of the covalent radius
_MULTI_OFFSET = 0.26          # half-separation (A) between parallel double/triple sticks
_BOND_LW = 2.2                # single-bond stick core width (points); higher orders thinner
_H_BOND_FACTOR = 0.62         # bonds to H drawn this fraction as thick (thinner than H ball)
_OUTLINE_EXTRA = 0.8          # dark halo width added around each stick core (points)
_ELEV, _AZIM = 20.0, 38.0     # default camera orientation
_BOND_ORDER_WIDTH = {1: 1.0, 2: 0.78, 3: 0.64}        # stick width ratio per bond order
_BOND_ORDER_SHIFTS = {1: (0.0,), 2: (-1.0, 1.0), 3: (-1.0, 0.0, 1.0)}

# Unit-sphere parametric mesh, reused (scaled/translated) for every atom.
_SPHERE_NU, _SPHERE_NV = 22, 16
_sphere_u = np.linspace(0.0, 2.0 * np.pi, _SPHERE_NU)
_sphere_v = np.linspace(0.0, np.pi, _SPHERE_NV)
_SPHERE_X = np.outer(np.cos(_sphere_u), np.sin(_sphere_v))
_SPHERE_Y = np.outer(np.sin(_sphere_u), np.sin(_sphere_v))
_SPHERE_Z = np.outer(np.ones_like(_sphere_u), np.cos(_sphere_v))


@dataclass(frozen=True)
class TrajectoryFrame:
    symbols: tuple[str, ...]
    coordinates: np.ndarray
    energy_kcal_mol: float | None = None


@dataclass(frozen=True)
class MoldenTrajectory:
    frames: tuple[TrajectoryFrame, ...]
    metrics: dict[str, list[float]]

    @property
    def energies(self) -> list[float | None]:
        return [frame.energy_kcal_mol for frame in self.frames]


def parse_molden_xyz_trajectory(path: str | Path) -> MoldenTrajectory:
    lines = Path(path).read_text().splitlines()
    frames: list[tuple[tuple[str, ...], np.ndarray]] = []
    metrics: dict[str, list[float]] = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "[GEOCONV]":
            i += 1
            break
        if line.startswith("["):
            i += 1
            continue
        try:
            natoms = int(line)
        except ValueError:
            i += 1
            continue

        i += 1
        if i < len(lines):
            i += 1

        symbols: list[str] = []
        coords: list[list[float]] = []
        for _ in range(natoms):
            parts = lines[i].split()
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            i += 1
        frames.append((tuple(symbols), np.asarray(coords, dtype=float)))

    current_metric: str | None = None
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith("["):
            break
        try:
            value = float(line)
        except ValueError:
            current_metric = line
            metrics[current_metric] = []
        else:
            if current_metric is None:
                raise ValueError(f"Metric value {value!r} found before metric label in {path}")
            metrics[current_metric].append(value)

    energies = metrics.get("energy", [])
    trajectory_frames: list[TrajectoryFrame] = []
    for index, (symbols, coords) in enumerate(frames):
        energy = energies[index] if index < len(energies) else None
        trajectory_frames.append(
            TrajectoryFrame(symbols=symbols, coordinates=coords, energy_kcal_mol=energy)
        )

    return MoldenTrajectory(frames=tuple(trajectory_frames), metrics=metrics)


def find_preferred_trajectory_source(run_dir: str | Path, run_id: int) -> Path:
    run_dir = Path(run_dir)
    candidates = [
        run_dir / f"opt_converged_{run_id:03d}.xyz",
        run_dir / f"grown_string1_{run_id:03d}.xyz",
        run_dir / f"grown_string_{run_id:03d}.xyz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    scratch_globs = [
        run_dir / "scratch" / f"opt_iters_{run_id:03d}_*.xyz",
        run_dir / "scratch" / f"growth_iters_{run_id:03d}_*.xyz",
    ]
    for pattern in scratch_globs:
        matches = sorted(pattern.parent.glob(pattern.name))
        if matches:
            return matches[-1]

    known = ", ".join(path.name for path in candidates)
    scratch_known = ", ".join(path.name for path in scratch_globs)
    raise FileNotFoundError(
        f"No trajectory file found in {run_dir}. "
        f"Looked for: {known}, and scratch snapshots matching: {scratch_known}"
    )


def resolve_trajectory_source(
    run_dir: str | Path,
    recorded_source: str | Path | None = None,
    run_id: int = 0,
) -> Path | None:
    """Locate a candidate's SE-GSM trajectory XYZ, robust to stale absolute paths.

    A summary's ``trajectory_source`` is an absolute path recorded on the compute
    node (e.g. ``/…/cscratch/jobs/<uuid>/…/opt_converged_000.xyz``). Once the run
    tree is copied elsewhere that path no longer resolves, which is why figures
    (``trajectory.gif``) silently went missing. This resolves the trajectory
    *locally*, in priority order:

    1. ``recorded_source`` as-is, then by basename in ``run_dir`` and
       ``run_dir/scratch`` (handles the copied/renamed tree);
    2. the conventional fully-optimized → grown-string filenames for this
       ``run_id``, then any run id, in ``run_dir`` and ``scratch``;
    3. partial ``growth_iters_*.xyz`` snapshots (failed/non-converged runs) —
       the most-grown string.

    Returns a :class:`pathlib.Path`, or ``None`` when nothing is found. Never
    raises, so callers can degrade to the static energy-profile fallback.
    """
    run_dir = Path(run_dir)
    search_dirs = [run_dir, run_dir / "scratch"]

    if recorded_source:
        name = Path(recorded_source).name
        for candidate in (run_dir / name, run_dir / "scratch" / name, Path(recorded_source)):
            if candidate.exists():
                return candidate

    exact = [
        f"opt_converged_{run_id:03d}.xyz",
        f"grown_string1_{run_id:03d}.xyz",
        f"grown_string_{run_id:03d}.xyz",
    ]
    globs = ["opt_converged_*.xyz", "grown_string1_*.xyz", "grown_string_*.xyz"]
    for name in exact:
        for search_dir in search_dirs:
            candidate = search_dir / name
            if candidate.exists():
                return candidate
    for pattern in globs:
        for search_dir in search_dirs:
            hits = sorted(search_dir.glob(pattern))
            if hits:
                return hits[0]

    for search_dir in search_dirs:
        hits = sorted(search_dir.glob(f"growth_iters_{run_id:03d}_*.xyz")) or sorted(
            search_dir.glob("growth_iters_*.xyz")
        )
        if hits:
            return hits[-1]
    return None


def bonds_with_order(
    frame: TrajectoryFrame, scale: float = BOND_SCALE
) -> list[tuple[int, int, int]]:
    """Detected bonds as ``(i, j, order)`` with ``order`` in {1, 2, 3}.

    Connectivity is the usual covalent-radius distance test; the order is then
    assigned from the bond length relative to the reference single/double/triple
    lengths for that element pair (see ``_BOND_REF``). Bonds to hydrogen and
    unlisted pairs are always single.
    """
    symbols = frame.symbols
    coords = frame.coordinates
    bonds: list[tuple[int, int, int]] = []
    for i, symbol_i in enumerate(symbols):
        radius_i = _COVALENT_RADII.get(symbol_i, _DEFAULT_RADIUS)
        for j in range(i + 1, len(symbols)):
            symbol_j = symbols[j]
            radius_j = _COVALENT_RADII.get(symbol_j, _DEFAULT_RADIUS)
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            if not (0.4 < distance <= scale * (radius_i + radius_j)):
                continue
            order = 1
            if "H" not in (symbol_i, symbol_j):
                ref = _BOND_REF.get(tuple(sorted((symbol_i, symbol_j))))
                if ref is not None:
                    single, double, triple = ref
                    if triple is not None and distance <= (double + triple) / 2.0:
                        order = 3
                    elif distance <= (single + double) / 2.0:
                        order = 2
            bonds.append((i, j, order))
    return bonds


def infer_bonds(frame: TrajectoryFrame, scale: float = BOND_SCALE) -> list[tuple[int, int]]:
    """Connectivity only (no bond order); see :func:`bonds_with_order`."""
    return [(i, j) for i, j, _order in bonds_with_order(frame, scale)]


def write_trajectory_sdf(
    trajectory: MoldenTrajectory,
    output_path: str | Path,
    *,
    case_name: str,
    reaction_label: str,
    formula: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[str] = []
    for frame_index, frame in enumerate(trajectory.frames):
        bonds = infer_bonds(frame)
        atom_lines = []
        for symbol, (x, y, z) in zip(frame.symbols, frame.coordinates):
            atom_lines.append(
                f"{x:>10.4f}{y:>10.4f}{z:>10.4f} {symbol:<3} 0  0  0  0  0  0  0  0  0  0  0  0"
            )
        bond_lines = [
            f"{a + 1:>3}{b + 1:>3}{1:>3}  0  0  0  0"
            for a, b in bonds
        ]
        record = [
            f"{case_name}__frame_{frame_index:03d}",
            "ReactIP",
            reaction_label,
            f"{len(frame.symbols):>3}{len(bonds):>3}  0  0  0  0            999 V2000",
            *atom_lines,
            *bond_lines,
            "M  END",
            ">  <CASE>",
            case_name,
            "",
            ">  <REACTION_LABEL>",
            reaction_label,
            "",
            ">  <FORMULA>",
            formula,
            "",
            ">  <FRAME_INDEX>",
            str(frame_index),
            "",
            ">  <RELATIVE_ENERGY_KCAL_MOL>",
            "None" if frame.energy_kcal_mol is None else f"{frame.energy_kcal_mol:.8f}",
            "",
            "$$$$",
        ]
        records.append("\n".join(record))

    output_path.write_text("\n".join(records) + "\n")
    return output_path


def _shift_color(color, fraction: float):
    """Lighten (fraction > 0) or darken (fraction < 0) a color toward white/black."""
    import matplotlib.colors as mcolors

    rgb = np.array(mcolors.to_rgb(color))
    if fraction < 0:
        return tuple(rgb * (1.0 + fraction))
    return tuple(rgb + (1.0 - rgb) * fraction)


def _camera_matrix(elev_deg: float, azim_deg: float) -> np.ndarray:
    """World->screen rotation, used to light spheres and offset multi-bonds."""
    elev, azim = np.radians(elev_deg), np.radians(azim_deg)
    ca, sa = np.cos(azim), np.sin(azim)
    rz = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])
    ce, se = np.cos(elev), np.sin(elev)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ce, se], [0.0, -se, ce]])
    return rx @ rz


def _sphere_facecolors(color, cam: np.ndarray) -> np.ndarray:
    """Per-face RGB for the unit-sphere mesh, lit from the viewer's upper-left."""
    import matplotlib.colors as mcolors

    rgb = np.array(mcolors.to_rgb(color))
    normals = np.stack(
        [_SPHERE_X[:-1, :-1], _SPHERE_Y[:-1, :-1], _SPHERE_Z[:-1, :-1]], axis=-1
    )
    in_view = normals @ cam.T
    light = np.array([-0.5, 0.5, 0.7])
    light /= np.linalg.norm(light)
    diffuse = np.clip(in_view @ light, 0.0, 1.0)
    shade = 0.32 + 0.68 * diffuse
    spec = diffuse ** 30
    return np.clip(
        rgb[None, None, :] * shade[..., None] + spec[..., None] * 0.85, 0.0, 1.0
    )


def _draw_stick(ax, start: np.ndarray, end: np.ndarray, color, linewidth: float) -> None:
    """One half-bond segment: thin dark outline + element-colored core."""
    ax.plot(
        [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
        color=_shift_color(color, -0.4), lw=linewidth + _OUTLINE_EXTRA,
        solid_capstyle="round",
    )
    ax.plot(
        [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
        color=color, lw=linewidth, solid_capstyle="round",
    )


def _render_structure_3d(
    ax,
    frame: TrajectoryFrame,
    *,
    center: np.ndarray,
    radius: float,
    elev: float = _ELEV,
    azim: float = _AZIM,
    atom_scale: float = _ATOM_SCALE,
    multi_offset: float = _MULTI_OFFSET,
    bond_lw: float = _BOND_LW,
    h_bond_factor: float = _H_BOND_FACTOR,
    show_labels: bool = True,
    label_h: bool = False,
) -> None:
    """Draw one frame as shaded ball-and-stick (with bond orders) into a 3D axes.

    Atoms are shaded spheres (``plot_surface``); bonds are split half/half by
    element; double / triple bonds are drawn as 2 / 3 parallel sticks offset in
    the screen plane. The familiar 3D box / panes / dotted grid / x-y-z axes are
    applied here too, so the camera and world extent stay fixed across frames.
    """
    import matplotlib.patheffects as pe

    symbols = frame.symbols
    coords = frame.coordinates
    cam = _camera_matrix(elev, azim)
    view_dir = cam[2, :]                          # world direction toward the viewer

    # Bonds first, so atom spheres sort in front of the sticks they connect.
    for i, j, order in bonds_with_order(frame):
        p, q = coords[i], coords[j]
        color_i = _ELEMENT_COLORS.get(symbols[i], _DEFAULT_COLOR)
        color_j = _ELEMENT_COLORS.get(symbols[j], _DEFAULT_COLOR)
        axis_vec = q - p
        norm = float(np.linalg.norm(axis_vec))
        axis_vec = axis_vec / norm if norm > 1e-6 else axis_vec
        perp = np.cross(axis_vec, view_dir)        # perpendicular to bond, in screen plane
        perp_norm = float(np.linalg.norm(perp))
        perp = perp / perp_norm if perp_norm > 1e-6 else np.array([0.0, 0.0, 1.0])
        linewidth = bond_lw * _BOND_ORDER_WIDTH[order]
        if "H" in (symbols[i], symbols[j]):        # keep C-H sticks thinner than the H ball
            linewidth *= h_bond_factor
        for shift in _BOND_ORDER_SHIFTS[order]:
            offset = perp * shift * multi_offset
            start, end = p + offset, q + offset
            mid = (start + end) / 2.0
            _draw_stick(ax, start, mid, color_i, linewidth)
            _draw_stick(ax, mid, end, color_j, linewidth)

    for index, symbol in enumerate(symbols):
        r = atom_scale * _COVALENT_RADII.get(symbol, _DEFAULT_RADIUS)
        x0, y0, z0 = coords[index]
        facecolors = _sphere_facecolors(_ELEMENT_COLORS.get(symbol, _DEFAULT_COLOR), cam)
        ax.plot_surface(
            _SPHERE_X * r + x0, _SPHERE_Y * r + y0, _SPHERE_Z * r + z0,
            facecolors=facecolors, rstride=1, cstride=1, linewidth=0,
            antialiased=True, shade=False, zsort="max",
        )
        if show_labels and (label_h or symbol != "H"):
            text = ax.text(
                x0, y0, z0, symbol, ha="center", va="center", zorder=1e4,
                fontsize=8.5 if symbol != "H" else 6.5, color="white", fontweight="bold",
            )
            text.set_path_effects([
                pe.withStroke(
                    linewidth=1.8,
                    foreground=_shift_color(_ELEMENT_COLORS.get(symbol, _DEFAULT_COLOR), -0.6),
                )
            ])

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    tick_offsets = np.linspace(-0.7 * radius, 0.7 * radius, 3)
    ax.set_xticks(np.round(center[0] + tick_offsets, 1))
    ax.set_yticks(np.round(center[1] + tick_offsets, 1))
    ax.set_zticks(np.round(center[2] + tick_offsets, 1))
    ax.set_xlabel("x [Å]", fontsize=9, labelpad=6)
    ax.set_ylabel("y [Å]", fontsize=9, labelpad=6)
    ax.set_zlabel("z [Å]", fontsize=9, labelpad=6)
    ax.tick_params(labelsize=8, pad=1, colors="#475569")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((0.80, 0.84, 0.89, 1.0))
        axis._axinfo["grid"].update(
            {"linewidth": 0.7, "linestyle": ":", "color": "#cbd5e1"}
        )


def render_trajectory_gif(
    trajectory: MoldenTrajectory,
    output_path: str | Path,
    *,
    title: str,
    duration: float = 0.85,
    elev: float = _ELEV,
    azim: float = _AZIM,
    atom_scale: float = _ATOM_SCALE,
    show_labels: bool = True,
    label_h: bool = False,
) -> Path:
    try:
        import imageio.v2 as imageio
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "GIF export requires the optional plotting dependencies: matplotlib and imageio."
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_coords = np.vstack([frame.coordinates for frame in trajectory.frames])
    minima = np.min(all_coords, axis=0)
    maxima = np.max(all_coords, axis=0)
    center = (minima + maxima) / 2.0
    radius = max(np.max(maxima - minima) / 2.0, 0.8) + 0.35
    energy_points = [
        (index, frame.energy_kcal_mol)
        for index, frame in enumerate(trajectory.frames)
        if frame.energy_kcal_mol is not None
    ]
    if energy_points:
        energy_indices = np.asarray([index + 1 for index, _ in energy_points], dtype=float)
        energy_values = np.asarray([energy for _, energy in energy_points], dtype=float)
        energy_padding = max((energy_values.max() - energy_values.min()) * 0.12, 8.0)
        energy_limits = (
            float(energy_values.min() - energy_padding),
            float(energy_values.max() + energy_padding),
        )
    else:
        energy_indices = np.asarray([], dtype=float)
        energy_values = np.asarray([], dtype=float)
        energy_limits = (-1.0, 1.0)

    frames_rgb = []
    for frame_index, frame in enumerate(trajectory.frames):
        fig = plt.figure(figsize=(9.2, 5.6), dpi=140)
        fig.patch.set_facecolor("#f8fafc")
        grid = fig.add_gridspec(
            nrows=2,
            ncols=2,
            width_ratios=(2.05, 1.55),
            height_ratios=(0.42, 1.18),
            wspace=0.18,
            hspace=0.10,
        )
        ax = fig.add_subplot(grid[:, 0], projection="3d")
        stats_ax = fig.add_subplot(grid[0, 1])
        energy_ax = fig.add_subplot(grid[1, 1])

        ax.set_facecolor("#f8fafc")
        stats_ax.set_facecolor("#eef2f7")
        energy_ax.set_facecolor("white")

        _render_structure_3d(
            ax,
            frame,
            center=center,
            radius=radius,
            elev=elev,
            azim=azim,
            atom_scale=atom_scale,
            show_labels=show_labels,
            label_h=label_h,
        )
        ax.set_title(title, fontsize=14, pad=16)

        energy_label = "n/a"
        if frame.energy_kcal_mol is not None:
            energy_label = f"{frame.energy_kcal_mol:.2f} kcal/mol"

        stats_ax.axis("off")
        stats_ax.text(
            0.06,
            0.80,
            "Trajectory",
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
            transform=stats_ax.transAxes,
        )
        stats_ax.text(
            0.06,
            0.58,
            f"Frame {frame_index + 1}/{len(trajectory.frames)}",
            fontsize=12,
            color="#0f172a",
            transform=stats_ax.transAxes,
        )
        stats_ax.text(
            0.06,
            0.37,
            f"Relative energy: {energy_label}",
            fontsize=11,
            color="#1e293b",
            transform=stats_ax.transAxes,
        )
        stats_ax.text(
            0.06,
            0.17,
            "x-axis in the plot: string node index.",
            fontsize=9,
            color="#475569",
            transform=stats_ax.transAxes,
        )

        energy_ax.set_title("Relative energy along string", fontsize=10, pad=8)
        energy_ax.set_xlabel("String node", fontsize=9)
        energy_ax.set_ylabel("Rel. E [kcal/mol]", fontsize=9)
        energy_ax.tick_params(labelsize=8, colors="#475569")
        energy_ax.axhline(0.0, color="#cbd5e1", linewidth=1.0, linestyle="--")
        for spine in energy_ax.spines.values():
            spine.set_color("#cbd5e1")
        if len(energy_indices) > 0:
            energy_ax.plot(
                energy_indices,
                energy_values,
                color="#475569",
                linewidth=1.8,
                marker="o",
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor="#475569",
            )
            if frame.energy_kcal_mol is not None:
                energy_ax.scatter(
                    [frame_index + 1],
                    [frame.energy_kcal_mol],
                    s=72,
                    color="#dc2626",
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=3,
                )
        energy_ax.set_xlim(0.7, len(trajectory.frames) + 0.3)
        energy_ax.set_ylim(*energy_limits)
        energy_ax.grid(True, linestyle=":", linewidth=0.7, color="#e2e8f0")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames_rgb.append(rgba[:, :, :3].copy())
        plt.close(fig)

    imageio.mimsave(output_path, frames_rgb, duration=duration, loop=0)
    return output_path


def export_trajectory_artifacts(
    trajectory_path: str | Path,
    output_dir: str | Path,
    *,
    case_name: str,
    reaction_label: str,
    formula: str,
) -> dict[str, str | int | list[float | None]]:
    trajectory_path = Path(trajectory_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory = parse_molden_xyz_trajectory(trajectory_path)
    sdf_path = write_trajectory_sdf(
        trajectory,
        output_dir / "trajectory.sdf",
        case_name=case_name,
        reaction_label=reaction_label,
        formula=formula,
    )
    warnings: list[str] = []
    gif_path: Path | None = None
    try:
        gif_path = render_trajectory_gif(
            trajectory,
            output_dir / "trajectory.gif",
            title=reaction_label,
        )
    except ModuleNotFoundError as exc:
        warnings.append(str(exc))
    return {
        "trajectory_source": str(trajectory_path),
        "frame_count": len(trajectory.frames),
        "energies": trajectory.energies,
        "artifact_paths": {
            "trajectory_sdf": str(sdf_path),
            "trajectory_gif": str(gif_path) if gif_path is not None else None,
        },
        "warnings": warnings,
    }


def write_summary_json(summary: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return output_path
