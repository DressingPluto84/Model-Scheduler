"""Run the bench over a matrix of workload shapes and chart the LLM results.

    python3 scripts/matrix.py run  results/matrix     # runs every cell, writes summary.csv and charts
    python3 scripts/matrix.py plot results/matrix     # redraws the charts from summary.csv

Axes: burstiness x tenant skew x LLM prompt length, each policy, several
seeds. Each prompt length runs at its own heavy load, about twice its peak
hour (where premium p99 time to first token is a few seconds), so the LLM
scheduler has requests from two or more tenants waiting on most picks and
its choice is what's being tested. For every run it records LLM time to
first token percentiles, the share of requests whose first token came within
the 2s target, full LLM goodput, and the share of contested picks, then
deletes the raw samples.
"""

import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BURSTS = [1, 2, 3]
SKEWS = [0, 1, 1.5]
# Prompt length -> (premium, free) tenants: about 2x the peak-hour load.
LOADS = {300: (76, 172), 1000: (40, 90), 2000: (26, 58)}
SEEDS = [1, 2, 3]
POLICIES = ["fcfs", "eevdf", "drr"]
COMMON = ["-sleeper=false", "-duration", "5m", "-llm-output-median", "130",
          "-premium-weight", "5000", "-free-weight", "1", "-policies", ",".join(POLICIES)]
PCTS = [50, 90, 99, 99.9]

FIELDS = ["input_median", "burst", "skew", "seed", "premium", "free", "policy", "tier",
          "contested", "ttft_slo", "goodput", "n"] + [f"p{p:g}" for p in PCTS]


def run_cell(out, bench, inp, burst, skew, seed):
    prem, free = LOADS[inp]
    name = f"in{inp}_b{burst:g}_s{skew:g}_seed{seed}"
    txt, samples = out / "runs" / f"{name}.txt", out / "runs" / f"{name}.csv"
    if txt.exists() and "Requests:" in txt.read_text():
        return name, None  # already done
    args = [str(bench), "-llm-input-median", str(inp), "-burst", str(burst), "-skew", str(skew),
            "-seed", str(seed), "-premium", str(prem), "-free", str(free), "-samples", str(samples)] + COMMON
    with open(txt, "w") as f:
        subprocess.run(args, stdout=f, stderr=subprocess.DEVNULL, check=True)
    rows = summarize(txt, samples, inp, burst, skew, seed, prem, free)
    samples.unlink()
    return name, rows


def summarize(txt, samples, inp, burst, skew, seed, prem, free):
    ttft = defaultdict(list)
    with open(samples, newline="") as f:
        for r in csv.DictReader(f):
            if r["model"] == "llm" and r["metric"] == "ttft":
                ttft[(r["policy"], r["tier"])].append(float(r["ms"]))
    goodput, ttft_slo, contested = {}, {}, {}
    section = ""
    for line in txt.read_text().splitlines():
        if line and not line.startswith(" "):
            section = line
            continue
        pcts = re.findall(r"(\d+)%", line)
        if section.startswith("Goodput"):
            m = re.match(r"\s+llm (premium|free)\s", line)
            if m:
                goodput.update({(p, m.group(1)): int(g) for p, g in zip(POLICIES, pcts)})
        elif section.startswith("LLM time to first token within SLO"):
            m = re.match(r"\s+(premium|free)\s", line)
            if m:
                ttft_slo.update({(p, m.group(1)): int(g) for p, g in zip(POLICIES, pcts)})
        elif section.startswith("Scheduler choice"):
            if re.match(r"\s+llm\s", line):
                contested.update({p: int(g) for p, g in zip(POLICIES, pcts)})
    rows = []
    for pol in POLICIES:
        for tier in ("premium", "free"):
            v = np.array(ttft[(pol, tier)])
            row = dict(input_median=inp, burst=burst, skew=skew, seed=seed, premium=prem, free=free,
                       policy=pol, tier=tier, contested=contested.get(pol, ""),
                       ttft_slo=ttft_slo.get((pol, tier), ""), goodput=goodput.get((pol, tier), ""), n=len(v))
            for p in PCTS:
                row[f"p{p:g}"] = round(float(np.percentile(v, p)), 1) if len(v) else ""
            rows.append(row)
    return rows


def run(out):
    out = Path(out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    bench = out / "bench"
    subprocess.run(["go", "build", "-o", str(bench), "./cmd/bench"], check=True)

    summary = out / "summary.csv"
    done = set()
    if summary.exists():
        with open(summary, newline="") as f:
            done = {(int(r["input_median"]), float(r["burst"]), float(r["skew"]), int(r["seed"]))
                    for r in csv.DictReader(f)}
    cells = [(i, b, s, seed) for i in LOADS for b in BURSTS for s in SKEWS for seed in SEEDS
             if (i, float(b), float(s), seed) not in done]
    print(f"{len(cells)} runs to go ({len(done)} already in {summary})")

    new = not summary.exists()
    with open(summary, "a", newline="") as f, ThreadPoolExecutor(max(1, (os.cpu_count() or 4) - 2)) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        futures = [pool.submit(run_cell, out, bench, *c) for c in cells]
        for i, fut in enumerate(futures, 1):
            name, rows = fut.result()
            if rows:
                w.writerows(rows)
                f.flush()
            print(f"  [{i}/{len(cells)}] {name}", flush=True)
    bench.unlink(missing_ok=True)
    plot(out)


# --- charts -----------------------------------------------------------------

COLORS = {"eevdf": "#2a78d6", "drr": "#eb6834", "fcfs": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(out):
    data = defaultdict(list)  # (inp, burst, skew, policy, tier, metric) -> values over seeds
    with open(Path(out) / "summary.csv", newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["input_median"]), float(r["burst"]), float(r["skew"]), r["policy"], r["tier"])
            for metric in ["contested", "ttft_slo", "goodput"] + [f"p{p:g}" for p in PCTS]:
                if r[metric] != "":
                    data[key + (metric,)].append(float(r[metric]))
    return data


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt

    data = load(out)
    cats = [(b, s) for b in BURSTS for s in SKEWS]
    labels = [f"burst {b:g}\nskew {s:g}" for b, s in cats]
    x = np.arange(len(cats))
    offsets = {p: (i - 1) * 0.22 for i, p in enumerate(POLICIES)}

    def figure(fname, title, rows, log, fmt_y):
        fig, axes = plt.subplots(len(rows), len(LOADS), figsize=(17, 3.6 * len(rows)),
                                 layout="constrained", squeeze=False, sharey="row")
        fig.suptitle(title, color=INK, fontsize=13, x=0.005, ha="left")
        for ri, (row_title, tier, metric) in enumerate(rows):
            for ci, inp in enumerate(LOADS):
                ax = axes[ri][ci]
                prem, free = LOADS[inp]
                for pol in POLICIES:
                    mean, lo, hi = [], [], []
                    for b, s in cats:
                        v = data.get((inp, float(b), float(s), pol, tier, metric), [])
                        mean.append(np.mean(v) if v else np.nan)
                        lo.append(np.min(v) if v else np.nan)
                        hi.append(np.max(v) if v else np.nan)
                    mean, lo, hi = map(np.array, (mean, lo, hi))
                    ax.errorbar(x + offsets[pol], mean, yerr=[mean - lo, hi - mean], fmt="o", ms=6,
                                color=COLORS[pol], ecolor=COLORS[pol], elinewidth=1.5, capsize=0,
                                label=pol if (ri, ci) == (0, 0) else None)
                if ri == 0:
                    cs = [v for k, vs in data.items() if k[0] == inp and k[5] == "contested" for v in vs]
                    ax.set_title(f"prompt median {inp} tokens · {prem} premium + {free} free\n"
                                 f"contested picks {min(cs):.0f}–{max(cs):.0f}%" if cs else
                                 f"prompt median {inp} tokens · {prem} premium + {free} free",
                                 loc="left", color=INK, fontsize=10)
                if ci == 0:
                    ax.set_ylabel(row_title, color=MUTED)
                if log:
                    ax.set_yscale("log")
                ax.yaxis.set_major_formatter(mt.FuncFormatter(fmt_y))
                ax.set_xticks(x, labels if ri == len(rows) - 1 else [""] * len(x), fontsize=8, color=MUTED)
                ax.grid(True, axis="y", color=GRID, linewidth=0.8)
                ax.set_axisbelow(True)
                ax.tick_params(colors=MUTED, labelsize=8)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
                for s in ("left", "bottom"):
                    ax.spines[s].set_color(GRID)
        if log:
            for ri, (_, tier, metric) in enumerate(rows):
                vals = [v for k, vs in data.items() if k[4] == tier and k[5] == metric for v in vs if v > 0]
                if vals:
                    axes[ri][0].set_ylim(min(vals) / 1.5, max(vals) * 1.5)
        fig.legend(loc="outside upper right", frameon=False, ncols=4, labelcolor=INK,
                   title="dot = mean over seeds, bar = min to max", title_fontsize=8, alignment="right")
        path = Path(out) / fname
        fig.savefig(path, dpi=130, facecolor="white")
        plt.close(fig)
        print(f"saved {path}")

    secs = lambda v, _: f"{v / 1000:g}s" if v >= 1000 else f"{v:g}ms"
    pct = lambda v, _: f"{v:g}%"
    figure("llm-ttft-slo.png", "LLM requests with first token within 2s",
           [("premium", "premium", "ttft_slo"), ("free", "free", "ttft_slo")], False, pct)
    figure("llm-ttft-premium.png", "LLM premium time to first token",
           [("p50", "premium", "p50"), ("p99", "premium", "p99"), ("p99.9", "premium", "p99.9")], True, secs)
    figure("llm-ttft-free.png", "LLM free time to first token",
           [("p50", "free", "p50"), ("p99", "free", "p99")], True, secs)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("run", "plot"):
        sys.exit(__doc__)
    {"run": run, "plot": plot}[sys.argv[1]](sys.argv[2])
