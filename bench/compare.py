#!/usr/bin/env python3
"""Compare two Google Benchmark JSON files (baseline vs current).

Usage:
    python bench/compare.py bench/baseline.json bench/results/current.json [--threshold 5]

Uses the *median* aggregate when the run had --benchmark_repetitions (recommended: 10),
otherwise the single value. Compares real_time and the gpu_us counter.
Exit code 1 if any metric regressed by more than max(--threshold, 3 x its coefficient of
variation) percent, so an agent can use it as a pass/fail gate. Run with repetitions so the cv
aggregate exists; otherwise the plain threshold (default 5) applies.
"""
import argparse
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out, cv = {}, {}
    for b in data["benchmarks"]:
        if b.get("aggregate_name") == "cv":
            cv[b["run_name"]] = {"time": b["real_time"], "gpu_us": b.get("gpu_us", 0.0)}
    has_median = any(b.get("aggregate_name") == "median" for b in data["benchmarks"])
    for b in data["benchmarks"]:
        if b.get("error_occurred"):
            continue
        if has_median:
            if b.get("aggregate_name") != "median":
                continue
            name = b["run_name"]
        else:
            if b.get("run_type") == "aggregate":
                continue
            name = b["name"]
        metrics = {"time": (b["real_time"], b["time_unit"])}
        if "gpu_us" in b:
            metrics["gpu_us"] = (b["gpu_us"], "us")
        out[name] = metrics
    return out, cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("current")
    ap.add_argument("--threshold", type=float, default=5.0)
    args = ap.parse_args()

    (base, base_cv), (cur, cur_cv) = load(args.baseline), load(args.current)
    regressions, rows = [], []
    for name in sorted(set(base) | set(cur)):
        if name not in base or name not in cur:
            rows.append((name, "-", "-", "-", "NEW" if name in cur else "REMOVED"))
            continue
        for metric, (bv, unit) in base[name].items():
            if metric not in cur[name]:
                continue
            cv = cur[name][metric][0]
            if bv <= 0:
                continue
            pct = (cv - bv) / bv * 100.0
            status = "ok"
            # Noise-aware: allowed slowdown = max(threshold, 3 x run-to-run cv of either run).
            noise = max(base_cv.get(name, {}).get(metric, 0.0), cur_cv.get(name, {}).get(metric, 0.0))
            limit = max(args.threshold, 3.0 * noise * 100.0)
            # gpu_us below 5 us is timer noise: only flag real regressions above that.
            if pct > limit and not (metric == "gpu_us" and cv < 5.0):
                status = "REGRESSED"
                regressions.append(name)
            elif pct < -args.threshold:
                status = "improved"
            rows.append((f"{name} [{metric}]", f"{bv:.3f} {unit}", f"{cv:.3f} {unit}",
                         f"{pct:+.1f}%", status))

    w = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'benchmark'.ljust(w)}  {'baseline':>16}  {'current':>16}  {'delta':>8}  status")
    print("-" * (w + 60))
    for r in rows:
        print(f"{r[0].ljust(w)}  {r[1]:>16}  {r[2]:>16}  {r[3]:>8}  {r[4]}")
    print()
    if regressions:
        print(f"FAIL: {len(set(regressions))} benchmark(s) regressed > {args.threshold}%")
        return 1
    print(f"PASS: no regression > {args.threshold}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
