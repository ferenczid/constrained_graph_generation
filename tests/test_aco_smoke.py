"""Smoke test: the ACO generator runs end to end on a tiny instance."""
import subprocess
import sys
from pathlib import Path

ACO_DIR = Path(__file__).resolve().parents[1] / "aco_experiments"


def test_aco_single_run(tmp_path):
    # The script expects these output subfolders to exist (the SLURM scripts create them).
    (tmp_path / "metrics").mkdir()
    (tmp_path / "graphs").mkdir()
    result = subprocess.run(
        [sys.executable, "aco_single_run.py",
         "--N", "40", "--M", "195", "--diam", "4", "--cc", "0.4",
         "--trials", "1", "--ants", "10", "--max_iter", "10",
         "--out_dir", str(tmp_path)],
        cwd=ACO_DIR, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    csv_files = list((tmp_path / "metrics").glob("*.csv"))
    assert len(csv_files) == 1, "expected one metrics file"
    assert csv_files[0].stat().st_size > 0
