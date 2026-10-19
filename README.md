# constrained_graph_generation

Generating graphs with a prescribed diameter and clustering coefficient at the same time, using ant colony optimisation (ACO) and Markov-chain sampling.

Code accompanying:

> D. Ferenczi, A. Grigoriev. **Constrained Graph Generation: Preserving Diameter and Clustering Coefficient Simultaneously.** ICCS 2026, Lecture Notes in Computer Science. https://doi.org/10.1007/978-3-032-29924-6_51

## Installation

Tested with Python 3.10 on Ubuntu.

```bash
git clone https://github.com/ferenczid/constrained_graph_generation.git
cd constrained_graph_generation
conda env create -f environment.yml
conda activate constrained_graph_generation
```

## Quick start

Generate a graph with 40 nodes, 195 edges, diameter 4 and clustering coefficient about 0.4. This is a small demonstration run (10 ants, 10 iterations) that takes about a second:

```bash
cd aco_experiments
mkdir -p out/metrics out/graphs
python aco_single_run.py --N 40 --M 195 --diam 4 --cc 0.4 --trials 1 --ants 10 --max_iter 10 --out_dir out
```

The script writes one CSV row of summary statistics (no header row) to `out/metrics/` and, if a valid graph is found, the graph as JSON to `out/graphs/`. This small run is stochastic: in 10 test runs it found a valid graph 7 times. A larger number of ants and iterations, as used in the paper, makes success more likely.

Scripts import helper modules from their own folder, so run them from inside that folder. Use `python <script>.py --help` to list the options.
## Repository structure

| Folder | Contents |
|---|---|
| `aco_experiments/` | The ACO generator (`aco_single_run.py`), a sensitivity study of its parameters, and SLURM scripts |
| `mcmc_experiments/` | Markov-chain sampling to generate graphs, seed generators and plotting scripts |
| `iccs_2026_experiments/` | Scripts and result files for the comparisons reported in the ICCS 2026 paper |

The `snellius_*.sh` scripts launch many independent runs in parallel on the Dutch national supercomputer Snellius. They load Snellius-specific modules and a personal virtual environment, so they need adapting to run elsewhere. The SLURM scripts load Python 3.13 on Snellius; the quick start above was tested with Python 3.10.

## Tests

```bash
pytest
```
This runs a smoke test of the ACO generator on a tiny instance (about a second). It checks that the script runs and writes its output, not that the results match the paper.
## Citation

If you use this code, please cite the paper above (see also `CITATION.cff`).

```bibtex
@inproceedings{ferenczi2026constrained,
  title     = {Constrained Graph Generation: Preserving Diameter and Clustering Coefficient Simultaneously},
  author    = {Ferenczi, D{\'a}vid and Grigoriev, Alexander},
  booktitle = {Computational Science -- ICCS 2026},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
  doi       = {10.1007/978-3-032-29924-6_51}
}
```

## License

MIT, see `LICENSE`.
