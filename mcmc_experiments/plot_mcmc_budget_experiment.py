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


def natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def adjacency_from_edges(n: int, edges: list[list[int]]) -> np.ndarray:
    adj = np.zeros((n, n), dtype=float)
    for u, v in edges:
        u = int(u)
        v = int(v)
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def laplacian_spectrum(n: int, edges: list[list[int]]) -> np.ndarray:
    adj = adjacency_from_edges(n, edges)
    lap = np.diag(adj.sum(axis=1)) - adj
    return np.linalg.eigvalsh(lap)


def load_initial_edges(result: dict[str, Any], result_path: Path) -> list[list[int]]:
    if "initial_edges" in result:
        return result["initial_edges"]

    input_seed = result.get("input_seed")
    if not input_seed:
        raise KeyError(
            f"{result_path} has neither 'initial_edges' nor 'input_seed'."
        )

    candidates = [
        Path(input_seed),
        result_path.parent / input_seed,
        result_path.parent.parent.parent / input_seed,
    ]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open() as handle:
                return json.load(handle)["edges"]

    raise FileNotFoundError(
        f"Could not locate the input seed referenced by {result_path}: "
        f"{input_seed}"
    )


def load_ensemble(directory: Path) -> dict[str, Any]:
    files = sorted(directory.glob("sample_*.json"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No sample_*.json files found in {directory}")

    final_spectra: list[np.ndarray] = []
    initial_spectra: list[np.ndarray] = []
    acceptance_rates: list[float] = []
    accepted_moves: list[int] = []
    records: list[dict[str, Any]] = []

    for path in files:
        with path.open() as handle:
            result = json.load(handle)

        if not result.get("final_valid", False):
            raise ValueError(f"Non-valid final graph found in {path}")

        n = int(result["N"])
        final_spectra.append(laplacian_spectrum(n, result["edges"]))
        initial_spectra.append(
            laplacian_spectrum(n, load_initial_edges(result, path))
        )

        diagnostics = result["mcmc_diagnostics"]
        acceptance_rates.append(float(diagnostics["acceptance_rate"]))
        accepted_moves.append(int(diagnostics["accepted"]))
        records.append(result)

    first = records[0]
    return {
        "records": records,
        "final_spectra": np.vstack(final_spectra),
        "initial_spectra": np.vstack(initial_spectra),
        "acceptance_rates": np.asarray(acceptance_rates, dtype=float),
        "accepted_moves": np.asarray(accepted_moves, dtype=float),
        "metadata": {
            "N": int(first["N"]),
            "M": int(first["M"]),
            "diam_min": int(first["diam_min"]),
            "diam_max": int(first["diam_max"]),
            "target_cc": float(first["target_cc"]),
            "n_iter": int(first["n_iter_requested"]),
        },
    }


def mean_pairwise_distance(spectra: np.ndarray) -> float:
    if spectra.shape[0] < 2:
        return math.nan
    return float(np.mean(pdist(spectra, metric="euclidean")))


def bootstrap_pairwise_mean(
    spectra: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    estimate = mean_pairwise_distance(spectra)
    n_graphs = spectra.shape[0]

    if n_graphs < 2 or n_bootstrap <= 0:
        return estimate, math.nan, math.nan

    boot = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        indices = rng.integers(0, n_graphs, size=n_graphs)
        boot[b] = mean_pairwise_distance(spectra[indices])

    low, high = np.quantile(boot, [0.025, 0.975])
    return estimate, float(low), float(high)


def discover_budget_dirs(setting_dir: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in setting_dir.glob("iter_*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"iter_(\d+)", path.name)
        if match:
            found[int(match.group(1))] = path
    return found


def add_existing_50000(
    budget_dirs: dict[int, Path],
    existing_root: Path | None,
) -> None:
    if existing_root is None:
        return
    if (
        (existing_root / "mcmc_fixed").is_dir()
        and (existing_root / "mcmc_multiseed").is_dir()
    ):
        budget_dirs[50000] = existing_root


def setting_label(metadata: dict[str, Any], name: str) -> str:
    dmin = metadata["diam_min"]
    dmax = metadata["diam_max"]
    d_text = f"{dmin}" if dmin == dmax else f"[{dmin},{dmax}]"
    return (
        f"{name}\n"
        f"$n={metadata['N']}$, $m={metadata['M']}$, "
        f"$D={d_text}$, $C^*={metadata['target_cc']:.2f}$"
    )


def analyse_setting(
    name: str,
    budget_dirs: dict[int, Path],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not budget_dirs:
        raise FileNotFoundError(f"No budget directories found for {name}")

    rows: list[dict[str, Any]] = []
    first_loaded: dict[str, Any] | None = None

    for budget in sorted(budget_dirs):
        root = budget_dirs[budget]

        if budget == 50000 and (root / "mcmc_fixed").is_dir():
            fixed_dir = root / "mcmc_fixed"
            multi_dir = root / "mcmc_multiseed"
        else:
            fixed_dir = root / "mcmc_fixed"
            multi_dir = root / "mcmc_multiseed"

        for strategy, directory in [
            ("fixed_seed", fixed_dir),
            ("multiple_aco_seeds", multi_dir),
        ]:
            loaded = load_ensemble(directory)
            if first_loaded is None:
                first_loaded = loaded

            estimate, low, high = bootstrap_pairwise_mean(
                loaded["final_spectra"],
                rng=rng,
                n_bootstrap=n_bootstrap,
            )

            rows.append({
                "setting": name,
                "budget": budget,
                "strategy": strategy,
                "num_chains": loaded["final_spectra"].shape[0],
                "mean_pairwise_final_distance": estimate,
                "ci_low": low,
                "ci_high": high,
                "mean_acceptance_rate": float(
                    np.mean(loaded["acceptance_rates"])
                ),
                "std_acceptance_rate": float(
                    np.std(loaded["acceptance_rates"], ddof=1)
                ),
                "mean_accepted_moves": float(
                    np.mean(loaded["accepted_moves"])
                ),
                "all_final_valid": all(
                    bool(record.get("final_valid", False))
                    for record in loaded["records"]
                ),
                "N": loaded["metadata"]["N"],
                "M": loaded["metadata"]["M"],
                "diam_min": loaded["metadata"]["diam_min"],
                "diam_max": loaded["metadata"]["diam_max"],
                "target_cc": loaded["metadata"]["target_cc"],
                "directory": str(directory),
            })

    assert first_loaded is not None

    # Add the N_iter=0 baseline using the initial graph ensembles from the
    # lowest-budget run. Fixed-seed diversity should be zero; multi-seed
    # diversity measures the diversity supplied directly by ACO.
    lowest_budget = min(budget_dirs)
    lowest_root = budget_dirs[lowest_budget]
    fixed_initial = load_ensemble(lowest_root / "mcmc_fixed")
    multi_initial = load_ensemble(lowest_root / "mcmc_multiseed")

    for strategy, loaded in [
        ("fixed_seed", fixed_initial),
        ("multiple_aco_seeds", multi_initial),
    ]:
        estimate, low, high = bootstrap_pairwise_mean(
            loaded["initial_spectra"],
            rng=rng,
            n_bootstrap=n_bootstrap,
        )
        rows.append({
            "setting": name,
            "budget": 0,
            "strategy": strategy,
            "num_chains": loaded["initial_spectra"].shape[0],
            "mean_pairwise_final_distance": estimate,
            "ci_low": low,
            "ci_high": high,
            "mean_acceptance_rate": math.nan,
            "std_acceptance_rate": math.nan,
            "mean_accepted_moves": 0.0,
            "all_final_valid": True,
            "N": loaded["metadata"]["N"],
            "M": loaded["metadata"]["M"],
            "diam_min": loaded["metadata"]["diam_min"],
            "diam_max": loaded["metadata"]["diam_max"],
            "target_cc": loaded["metadata"]["target_cc"],
            "directory": "initial seed ensemble",
        })

    return rows, first_loaded["metadata"]


def plot_diversity(
    all_rows: list[dict[str, Any]],
    metadata_by_setting: dict[str, dict[str, Any]],
    out_dir: Path,
) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=False)

    for ax, setting, title in [
        (axes[0], "strict", "Restrictive setting"),
        (axes[1], "loose", "Less restrictive setting"),
    ]:
        for strategy, label in [
            ("fixed_seed", "Fixed seed"),
            ("multiple_aco_seeds", "Multiple ACO seeds"),
        ]:
            rows = sorted(
                [
                    row for row in all_rows
                    if row["setting"] == setting
                    and row["strategy"] == strategy
                ],
                key=lambda row: row["budget"],
            )
            x = np.array([row["budget"] for row in rows], dtype=float)
            y = np.array(
                [row["mean_pairwise_final_distance"] for row in rows],
                dtype=float,
            )
            low = np.array([row["ci_low"] for row in rows], dtype=float)
            high = np.array([row["ci_high"] for row in rows], dtype=float)

            line, = ax.plot(x, y, marker="o", linewidth=2.0, label=label)
            ax.fill_between(
                x,
                low,
                high,
                alpha=0.18,
                color=line.get_color(),
            )

        ax.set_xscale("symlog", linthresh=500, base=10)
        budgets = sorted({
            int(row["budget"])
            for row in all_rows
            if row["setting"] == setting
        })
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{budget:,}" for budget in budgets])
        ax.set_xlabel("Number of raw MCMC proposals")
        ax.set_ylabel("Mean pairwise spectral distance")
        ax.set_title(setting_label(metadata_by_setting[setting], title))
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(
        out_dir / "mcmc_diversity_vs_budget.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / "mcmc_diversity_vs_budget.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_acceptance(
    all_rows: list[dict[str, Any]],
    metadata_by_setting: dict[str, dict[str, Any]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=False)

    for ax, setting, title in [
        (axes[0], "strict", "Restrictive setting"),
        (axes[1], "loose", "Less restrictive setting"),
    ]:
        for strategy, label in [
            ("fixed_seed", "Fixed seed"),
            ("multiple_aco_seeds", "Multiple ACO seeds"),
        ]:
            rows = sorted(
                [
                    row for row in all_rows
                    if row["setting"] == setting
                    and row["strategy"] == strategy
                    and row["budget"] > 0
                ],
                key=lambda row: row["budget"],
            )
            x = np.array([row["budget"] for row in rows], dtype=float)
            y = np.array(
                [row["mean_acceptance_rate"] for row in rows],
                dtype=float,
            )
            yerr = np.array(
                [row["std_acceptance_rate"] for row in rows],
                dtype=float,
            )

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=label,
            )

        ax.set_xscale("log")
        budgets = sorted({
            int(row["budget"])
            for row in all_rows
            if row["setting"] == setting and row["budget"] > 0
        })
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{budget:,}" for budget in budgets])
        ax.set_xlabel("Number of raw MCMC proposals")
        ax.set_ylabel("Acceptance rate")
        ax.set_title(setting_label(metadata_by_setting[setting], title))
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(
        out_dir / "mcmc_acceptance_vs_budget.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / "mcmc_acceptance_vs_budget.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    path = out_dir / "mcmc_budget_summary.csv"
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(
            rows,
            key=lambda row: (
                row["setting"],
                int(row["budget"]),
                row["strategy"],
            ),
        ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("mcmc_budget_experiment"),
    )
    parser.add_argument(
        "--strict_50000_dir",
        type=Path,
        default=Path(
            "results_strict_N40_M78_D12_CC0.35_mh_iter50000"
        ),
    )
    parser.add_argument(
        "--loose_50000_dir",
        type=Path,
        default=Path(
            "results_loose_N40_M195_D4_CC0.4_mh_iter50000"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("mcmc_budget_plots"),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of graph-level bootstrap replicates.",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=20260713,
        help=(
            "Seed used only for reproducible bootstrap confidence intervals; "
            "it does not affect the MCMC experiment."
        ),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)

    strict_dirs = discover_budget_dirs(args.root / "strict")
    loose_dirs = discover_budget_dirs(args.root / "loose")

    add_existing_50000(strict_dirs, args.strict_50000_dir)
    add_existing_50000(loose_dirs, args.loose_50000_dir)

    strict_rows, strict_metadata = analyse_setting(
        "strict",
        strict_dirs,
        rng=rng,
        n_bootstrap=args.bootstrap,
    )
    loose_rows, loose_metadata = analyse_setting(
        "loose",
        loose_dirs,
        rng=rng,
        n_bootstrap=args.bootstrap,
    )

    rows = strict_rows + loose_rows
    metadata = {
        "strict": strict_metadata,
        "loose": loose_metadata,
    }

    plot_diversity(rows, metadata, args.out_dir)
    plot_acceptance(rows, metadata, args.out_dir)
    write_summary(rows, args.out_dir)

    print("Wrote:")
    print(args.out_dir / "mcmc_diversity_vs_budget.pdf")
    print(args.out_dir / "mcmc_diversity_vs_budget.png")
    print(args.out_dir / "mcmc_acceptance_vs_budget.pdf")
    print(args.out_dir / "mcmc_acceptance_vs_budget.png")
    print(args.out_dir / "mcmc_budget_summary.csv")


if __name__ == "__main__":
    main()
