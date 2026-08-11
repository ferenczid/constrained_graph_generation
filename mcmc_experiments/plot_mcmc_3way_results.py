#!/usr/bin/env python3




from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde


TREATMENTS = (
    ("fixed_seed", "Fixed ACO seed", "mcmc_fixed"),
    ("multiple_aco_seeds", "Multiple ACO seeds", "mcmc_multiseed"),
    (
        "multiple_random_layered_seeds",
        "Multiple random-layered seeds",
        "mcmc_random_layered",
    ),
)


def natural_key(path: Path) -> list[Any]:
    """Sort sample_2.json before sample_10.json."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def adjacency_from_edges(n: int, edges: list[list[int]]) -> np.ndarray:
    adj = np.zeros((n, n), dtype=float)

    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"Malformed edge: {edge}")

        u, v = int(edge[0]), int(edge[1])

        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"Edge {(u, v)} is outside vertex range 0,...,{n - 1}.")
        if u == v:
            raise ValueError(f"Self-loop found: {(u, v)}")

        adj[u, v] = 1.0
        adj[v, u] = 1.0

    return adj


def laplacian_spectrum(n: int, edges: list[list[int]]) -> np.ndarray:
    """Eigenvalues of the combinatorial Laplacian L = D - A."""
    adj = adjacency_from_edges(n, edges)
    laplacian = np.diag(adj.sum(axis=1)) - adj
    return np.linalg.eigvalsh(laplacian)


def spectral_distance_from_spectra(
    spectrum_a: np.ndarray,
    spectrum_b: np.ndarray,
) -> float:
    return float(np.linalg.norm(spectrum_a - spectrum_b))


def candidate_seed_paths(input_seed: str, result_path: Path) -> list[Path]:
    """
    Return plausible paths for an input seed.

    The MCMC wrapper may store an absolute path, a path relative to the
    submission directory, or a path relative to the sample JSON.
    """
    raw = Path(input_seed)
    candidates = [raw]

    if not raw.is_absolute():
        candidates.extend(
            [
                result_path.parent / raw,
                result_path.parent.parent / raw,
                result_path.parent.parent.parent / raw,
            ]
        )

    # Preserve order while removing duplicates.
    unique: list[Path] = []
    seen: set[str] = set()

    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)

    return unique


def load_initial_edges(
    result: dict[str, Any],
    result_path: Path,
) -> list[list[int]]:
    """
    Prefer initial_edges saved directly in the MCMC result.

    Fall back to the input seed JSON for compatibility with older outputs.
    """
    if "initial_edges" in result:
        return result["initial_edges"]

    input_seed = result.get("input_seed")
    if not input_seed:
        raise KeyError(
            f"{result_path} has neither 'initial_edges' nor 'input_seed'."
        )

    tried: list[str] = []

    for seed_path in candidate_seed_paths(str(input_seed), result_path):
        tried.append(str(seed_path))

        if seed_path.exists():
            with seed_path.open() as handle:
                seed_data = json.load(handle)

            if "edges" not in seed_data:
                raise KeyError(f"Seed file {seed_path} contains no 'edges' field.")

            return seed_data["edges"]

    raise FileNotFoundError(
        f"Could not locate input seed for {result_path}. Tried: {tried}"
    )


def load_treatment(directory: Path) -> dict[str, Any]:
    files = sorted(directory.glob("sample_*.json"), key=natural_key)

    if not files:
        raise FileNotFoundError(f"No sample_*.json files found in {directory}")

    records: list[dict[str, Any]] = []
    final_spectra: list[np.ndarray] = []
    initial_spectra: list[np.ndarray] = []

    for path in files:
        with path.open() as handle:
            result = json.load(handle)

        if not result.get("final_valid", False):
            raise ValueError(f"Non-valid final graph found in {path}")

        n = int(result["N"])
        final_edges = result["edges"]
        initial_edges = load_initial_edges(result, path)

        final_spectra.append(laplacian_spectrum(n, final_edges))
        initial_spectra.append(laplacian_spectrum(n, initial_edges))
        records.append(result)

    final_spectra_array = np.vstack(final_spectra)
    initial_spectra_array = np.vstack(initial_spectra)

    pairwise_final = pdist(final_spectra_array, metric="euclidean")

    drift = np.array(
        [
            spectral_distance_from_spectra(initial, final)
            for initial, final in zip(initial_spectra_array, final_spectra_array)
        ],
        dtype=float,
    )

    diagnostics = [record["mcmc_diagnostics"] for record in records]

    acceptance_rates = np.array(
        [float(diagnostic["acceptance_rate"]) for diagnostic in diagnostics],
        dtype=float,
    )
    accepted_moves = np.array(
        [int(diagnostic["accepted"]) for diagnostic in diagnostics],
        dtype=float,
    )
    proposal_attempts = np.array(
        [int(diagnostic["proposal_attempts"]) for diagnostic in diagnostics],
        dtype=float,
    )
    runtimes = np.array(
        [float(record.get("runtime_seconds", np.nan)) for record in records],
        dtype=float,
    )

    first = records[0]
    metadata = {
        "N": int(first["N"]),
        "M": int(first["M"]),
        "diam_min": int(first["diam_min"]),
        "diam_max": int(first["diam_max"]),
        "target_cc": float(first["target_cc"]),
        "cc_tol": float(first["cc_tol"]),
        "n_iter": int(first["n_iter_requested"]),
    }

    return {
        "directory": str(directory),
        "records": records,
        "metadata": metadata,
        "pairwise_final": pairwise_final,
        "drift": drift,
        "acceptance_rates": acceptance_rates,
        "accepted_moves": accepted_moves,
        "proposal_attempts": proposal_attempts,
        "runtimes": runtimes,
    }


def validate_metadata(
    setting_name: str,
    treatments: list[dict[str, Any]],
) -> None:
    """Ensure all treatments within one panel use the same experiment."""
    reference = treatments[0]["metadata"]

    for treatment in treatments[1:]:
        current = treatment["metadata"]

        if current != reference:
            raise ValueError(
                f"Metadata mismatch in {setting_name}:\n"
                f"reference={reference}\n"
                f"current={current}\n"
                f"directory={treatment['directory']}"
            )


def finite_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "q25": math.nan,
            "q75": math.nan,
            "min": math.nan,
            "max": math.nan,
        }

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def draw_density(
    ax: plt.Axes,
    values: np.ndarray,
    label: str,
    grid: np.ndarray,
) -> Any:
    """
    Draw a KDE and return the line object.

    The returned line color is reused for the corresponding dashed mean line.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError(f"No finite values available for treatment: {label}")

    if np.std(values) < 1e-12:
        return ax.axvline(values[0], linewidth=2.0, label=label)

    kde = gaussian_kde(values)
    density = kde(grid)

    line, = ax.plot(grid, density, linewidth=2.0, label=label)
    ax.fill_between(
        grid,
        density,
        alpha=0.16,
        color=line.get_color(),
    )

    return line


def setting_title(name: str, metadata: dict[str, Any]) -> str:
    dmin = metadata["diam_min"]
    dmax = metadata["diam_max"]
    diameter_text = f"{dmin}" if dmin == dmax else f"[{dmin},{dmax}]"

    return (
        f"{name}\n"
        f"$n={metadata['N']}$, $m={metadata['M']}$, "
        f"$D={diameter_text}$, $C^*={metadata['target_cc']:.2f}$"
    )


def create_figure(
    strict_data: dict[str, dict[str, Any]],
    loose_data: dict[str, dict[str, Any]],
    out_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    panels = [
        ("Restrictive setting", strict_data),
        ("Less restrictive setting", loose_data),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=False)

    for ax, (setting_name, treatment_data) in zip(axes, panels):
        ordered = [
            (
                strategy_key,
                display_label,
                treatment_data[strategy_key],
            )
            for strategy_key, display_label, _ in TREATMENTS
        ]

        validate_metadata(
            setting_name,
            [data for _, _, data in ordered],
        )

        combined = np.concatenate(
            [data["pairwise_final"] for _, _, data in ordered]
        )
        finite = combined[np.isfinite(combined)]

        if finite.size == 0:
            raise ValueError(
                f"No finite pairwise distances for {setting_name}"
            )

        xmin = max(0.0, float(np.min(finite)))
        xmax = float(np.max(finite))
        span = max(xmax - xmin, 1e-6)
        grid = np.linspace(
            max(0.0, xmin - 0.06 * span),
            xmax + 0.08 * span,
            500,
        )

        for _, display_label, data in ordered:
            values = data["pairwise_final"]
            line = draw_density(ax, values, display_label, grid)

            # Match the dashed mean line to its density curve.
            ax.axvline(
                np.mean(values),
                color=line.get_color(),
                linestyle="--",
                linewidth=1.2,
                alpha=0.95,
            )

        reference_metadata = ordered[0][2]["metadata"]
        ax.set_title(setting_title(setting_name, reference_metadata))
        ax.set_xlabel("Pairwise spectral distance between final graphs")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    fig.tight_layout()

    pdf_path = out_dir / "mcmc_3way_final_ensemble_diversity.pdf"
    png_path = out_dir / "mcmc_3way_final_ensemble_diversity.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summary_row(
    setting: str,
    strategy: str,
    display_label: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    pairwise = finite_summary(data["pairwise_final"])
    drift = finite_summary(data["drift"])
    acceptance = finite_summary(data["acceptance_rates"])
    accepted = finite_summary(data["accepted_moves"])
    runtime = finite_summary(data["runtimes"])

    metadata = data["metadata"]

    return {
        "setting": setting,
        "strategy": strategy,
        "display_label": display_label,
        "N": metadata["N"],
        "M": metadata["M"],
        "diam_min": metadata["diam_min"],
        "diam_max": metadata["diam_max"],
        "target_cc": metadata["target_cc"],
        "cc_tol": metadata["cc_tol"],
        "n_iter": metadata["n_iter"],
        "num_chains": len(data["records"]),
        "num_pairwise_distances": len(data["pairwise_final"]),
        "pairwise_final_mean": pairwise["mean"],
        "pairwise_final_median": pairwise["median"],
        "pairwise_final_std": pairwise["std"],
        "pairwise_final_q25": pairwise["q25"],
        "pairwise_final_q75": pairwise["q75"],
        "pairwise_final_min": pairwise["min"],
        "pairwise_final_max": pairwise["max"],
        "drift_mean": drift["mean"],
        "drift_median": drift["median"],
        "acceptance_rate_mean": acceptance["mean"],
        "acceptance_rate_std": acceptance["std"],
        "accepted_moves_mean": accepted["mean"],
        "accepted_moves_std": accepted["std"],
        "runtime_seconds_mean": runtime["mean"],
        "runtime_seconds_std": runtime["std"],
        "all_final_valid": all(
            bool(record.get("final_valid", False))
            for record in data["records"]
        ),
        "directory": data["directory"],
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return math.nan
    if abs(denominator) < 1e-15:
        return math.nan
    return numerator / denominator


def write_summary(
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    csv_path = out_dir / "mcmc_3way_summary.csv"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_key = {
        (row["setting"], row["strategy"]): row
        for row in rows
    }

    txt_path = out_dir / "mcmc_3way_summary.txt"

    with txt_path.open("w") as handle:
        for row in rows:
            handle.write(
                f"{row['setting']} / {row['display_label']}\n"
                f"  chains: {row['num_chains']}\n"
                f"  pairwise distances: {row['num_pairwise_distances']}\n"
                f"  mean pairwise final distance: "
                f"{row['pairwise_final_mean']:.6f}\n"
                f"  median pairwise final distance: "
                f"{row['pairwise_final_median']:.6f}\n"
                f"  standard deviation: "
                f"{row['pairwise_final_std']:.6f}\n"
                f"  mean seed-to-final drift: "
                f"{row['drift_mean']:.6f}\n"
                f"  mean acceptance rate: "
                f"{row['acceptance_rate_mean']:.6f}\n"
                f"  mean accepted moves: "
                f"{row['accepted_moves_mean']:.2f}\n"
                f"  all final graphs valid: "
                f"{row['all_final_valid']}\n\n"
            )

        handle.write("Mean pairwise-diversity ratios\n")

        for setting in ("restrictive", "less_restrictive"):
            fixed = by_key[(setting, "fixed_seed")]["pairwise_final_mean"]
            multi_aco = by_key[
                (setting, "multiple_aco_seeds")
            ]["pairwise_final_mean"]
            random_layered = by_key[
                (setting, "multiple_random_layered_seeds")
            ]["pairwise_final_mean"]

            handle.write(
                f"  {setting}\n"
                f"    multiple ACO / fixed ACO: "
                f"{safe_ratio(multi_aco, fixed):.6f}\n"
                f"    random layered / fixed ACO: "
                f"{safe_ratio(random_layered, fixed):.6f}\n"
                f"    multiple ACO / random layered: "
                f"{safe_ratio(multi_aco, random_layered):.6f}\n"
            )


def load_setting(root: Path) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}

    for strategy_key, _, subdirectory in TREATMENTS:
        treatment_dir = root / subdirectory

        if not treatment_dir.exists():
            raise FileNotFoundError(
                f"Missing treatment directory: {treatment_dir}"
            )

        data[strategy_key] = load_treatment(treatment_dir)

    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict_dir",
        type=Path,
        default=Path(
            "results_3way_strict_N50_M245_D19_CC0.9_mh_iter5000"
        ),
    )
    parser.add_argument(
        "--loose_dir",
        type=Path,
        default=Path(
            "results_3way_loose_N50_M490_D4_CC0.4_mh_iter5000"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("mcmc_plots_3way"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    strict_data = load_setting(args.strict_dir)
    loose_data = load_setting(args.loose_dir)

    create_figure(
        strict_data=strict_data,
        loose_data=loose_data,
        out_dir=args.out_dir,
    )

    rows: list[dict[str, Any]] = []

    for setting_name, setting_data in (
        ("restrictive", strict_data),
        ("less_restrictive", loose_data),
    ):
        for strategy_key, display_label, _ in TREATMENTS:
            rows.append(
                summary_row(
                    setting=setting_name,
                    strategy=strategy_key,
                    display_label=display_label,
                    data=setting_data[strategy_key],
                )
            )

    write_summary(rows, args.out_dir)

    print("Wrote:")
    print(args.out_dir / "mcmc_3way_final_ensemble_diversity.pdf")
    print(args.out_dir / "mcmc_3way_final_ensemble_diversity.png")
    print(args.out_dir / "mcmc_3way_summary.csv")
    print(args.out_dir / "mcmc_3way_summary.txt")


if __name__ == "__main__":
    main()
