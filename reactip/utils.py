"""Trajectory parsing and export helpers for ReactIP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
}

_ELEMENT_COLORS = {
    "H": "#f3f4f6",
    "C": "#1f2937",
    "N": "#2563eb",
    "O": "#dc2626",
    "F": "#10b981",
    "S": "#d97706",
    "Cl": "#16a34a",
    "Br": "#92400e",
}

_ELEMENT_SIZES = {
    "H": 120,
    "C": 180,
    "N": 200,
    "O": 200,
    "F": 220,
    "S": 260,
    "Cl": 320,
    "Br": 360,
}


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


def infer_bonds(frame: TrajectoryFrame, scale: float = 1.20) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for i, symbol_i in enumerate(frame.symbols):
        radius_i = _COVALENT_RADII.get(symbol_i, 0.77)
        for j in range(i + 1, len(frame.symbols)):
            symbol_j = frame.symbols[j]
            radius_j = _COVALENT_RADII.get(symbol_j, 0.77)
            cutoff = scale * (radius_i + radius_j)
            distance = float(np.linalg.norm(frame.coordinates[i] - frame.coordinates[j]))
            if distance <= cutoff:
                bonds.append((i, j))
    return bonds


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


def render_trajectory_gif(
    trajectory: MoldenTrajectory,
    output_path: str | Path,
    *,
    title: str,
    duration: float = 0.85,
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

        bonds = infer_bonds(frame)
        for a, b in bonds:
            xyz = frame.coordinates[[a, b]]
            ax.plot(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                color="#94a3b8",
                linewidth=2.6,
                alpha=0.95,
            )

        for atom_index, symbol in enumerate(frame.symbols):
            coord = frame.coordinates[atom_index]
            ax.scatter(
                coord[0],
                coord[1],
                coord[2],
                s=_ELEMENT_SIZES.get(symbol, 180) * 1.15,
                c=_ELEMENT_COLORS.get(symbol, "#6b7280"),
                edgecolors="black",
                linewidths=0.7,
                depthshade=True,
            )

        ax.view_init(elev=20, azim=38)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        tick_offsets = np.linspace(-0.7 * radius, 0.7 * radius, 3)
        ax.set_xticks(np.round(center[0] + tick_offsets, 1))
        ax.set_yticks(np.round(center[1] + tick_offsets, 1))
        ax.set_zticks(np.round(center[2] + tick_offsets, 1))
        ax.set_xlabel("x [A]", fontsize=9, labelpad=6)
        ax.set_ylabel("y [A]", fontsize=9, labelpad=6)
        ax.set_zlabel("z [A]", fontsize=9, labelpad=6)
        ax.tick_params(labelsize=8, pad=1, colors="#475569")
        ax.set_title(title, fontsize=14, pad=16)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((0.80, 0.84, 0.89, 1.0))
            axis._axinfo["grid"].update(
                {"linewidth": 0.7, "linestyle": ":", "color": "#cbd5e1"}
            )

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
