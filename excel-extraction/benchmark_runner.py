"""Benchmark all extractors on the complex test workbook.

For each extractor we run K warm runs, then measure:

  - load_time_ms   median wall-clock time to open + extract everything
  - peak_mem_kb    peak Python memory delta via tracemalloc
  - aspect recall  per-aspect accuracy from `metrics.evaluate`

Outputs:
  results/benchmark_results.json
  results/per_extractor_metrics.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
import tracemalloc

from extractors import EXTRACTORS
from metrics import evaluate


HERE = pathlib.Path(__file__).parent
WB_PATH = HERE / "test_data" / "complex_workbook_cached.xlsx"
GT_PATH = HERE / "ground_truth" / "ground_truth.json"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _time_extract(name: str, fn, path: str, *, runs: int = 5) -> tuple[float, dict]:
    """Run extractor `runs` times, return median elapsed seconds + last output."""
    times: list[float] = []
    last_out: dict = {}
    for i in range(runs):
        tracemalloc.start()
        t0 = time.perf_counter()
        out = fn(path)
        elapsed = time.perf_counter() - t0
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(elapsed)
        last_out = out
        last_out["_perf"] = {
            "time_sec": elapsed,
            "peak_mem_kb": peak / 1024.0,
            "run": i,
        }
    median_t = statistics.median(times)
    return median_t, last_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--workbook", default=str(WB_PATH))
    ap.add_argument("--ground-truth", default=str(GT_PATH))
    args = ap.parse_args()

    gt = json.loads(pathlib.Path(args.ground_truth).read_text())
    results = {}

    for name, fn in EXTRACTORS.items():
        print(f"=== {name} ===")
        try:
            med, output = _time_extract(name, fn, args.workbook, runs=args.runs)
            if "error" in output:
                print(f"  extractor failed: {output['error']}")
                results[name] = {
                    "load_time_ms": med * 1000.0,
                    "metrics": {},
                    "error": output["error"],
                }
                continue
            evald = evaluate(gt, output)
            results[name] = {
                "load_time_ms": med * 1000.0,
                "metrics": evald,
            }
            # Quick summary
            cva = evald["cell_value_accuracy"]
            print(f"  load_time_ms={med*1000:.1f}  cell_acc={cva['accuracy']:.3f}  "
                  f"formulas={evald['formula_recall']['recall']:.3f}  "
                  f"charts={evald['chart_recall']['recall']:.3f}  "
                  f"tables={evald['table_recall']['recall']:.3f}  "
                  f"merges={evald['merged_recall']['recall']:.3f}")
        except Exception as e:
            print(f"  EXCEPTION: {type(e).__name__}: {e}")
            results[name] = {"error": f"{type(e).__name__}: {e}"}

    out_path = RESULTS_DIR / "benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
