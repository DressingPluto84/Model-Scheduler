"""Plot the latency distribution of one (model, tier) pair across policies.

    go run ./cmd/bench -premium 24 -free 54 -sleeper=false -samples results/samples.csv
    python3 scripts/dist.py results/samples.csv --model llm --tier premium

Prints a percentile table and saves a PNG with two views:
  left   histogram (log-spaced bins), where most requests land
  right  tail: share of requests slower than x, on a log scale, so p90, p99
         and p99.9 are the points where a curve crosses 10%, 1% and 0.1%

The metric defaults to time to first token for the LLM and end-to-end latency
for embed and rerank (--metric itl for the LLM's inter-token gaps). Requests
still waiting when the run ended have no sample, so tails under overload are
if anything understated.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Each policy keeps its color in every chart (reference categorical palette).
POLICY_COLORS = {"eevdf": "#2a78d6", "drr": "#eb6834", "fcfs": "#1baf7a"}
EXTRA_COLORS = ["#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

PERCENTILES = [50, 90, 95, 99, 99.9]


def load(path, model, tier, metric):
    by_policy = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] == model and row["tier"] == tier and row["metric"] == metric:
                by_policy[row["policy"]].append(float(row["ms"]))
    return {p: np.array(v) for p, v in by_policy.items()}


def fmt(ms):
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.0f}ms" if ms >= 10 else f"{ms:.1f}ms"


def table(data):
    head = ["policy", "n", "mean"] + [f"p{p:g}" for p in PERCENTILES] + ["max"]
    rows = []
    for p, v in data.items():
        rows.append([p, str(len(v)), fmt(v.mean())] + [fmt(np.percentile(v, q)) for q in PERCENTILES] + [fmt(v.max())])
    widths = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    for r in [head] + rows:
        print("  ".join(c.rjust(w) for c, w in zip(r, widths)))


def plot(data, title, out, slo):
    colors = {}
    extra = iter(EXTRA_COLORS)
    for p in data:
        colors[p] = POLICY_COLORS.get(p) or next(extra)

    lo = max(min(v.min() for v in data.values()), 0.1)
    hi = max(v.max() for v in data.values())
    bins = np.logspace(np.log10(lo), np.log10(hi * 1.01), 60)

    fig, (ax_h, ax_t) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")

    for p, v in data.items():
        w = np.full(len(v), 100 / len(v))
        ax_h.hist(v, bins=bins, weights=w, histtype="step", linewidth=2, color=colors[p], label=p)

        xs = np.sort(v)
        above = 100 * (1 - np.arange(1, len(xs) + 1) / len(xs))
        keep = above > 0
        ax_t.step(xs[keep], above[keep], where="post", linewidth=2, color=colors[p], label=p)
        # Direct label at the end of each tail curve.
        ax_t.annotate(p, (xs[keep][-1], above[keep][-1]), xytext=(4, 0), textcoords="offset points",
                      color=MUTED, fontsize=9, va="center")

    for q, label in [(10, "p90"), (1, "p99"), (0.1, "p99.9")]:
        ax_t.axhline(q, color=GRID, linewidth=1, zorder=0)
        ax_t.annotate(label, (0, q), xycoords=("axes fraction", "data"), xytext=(4, 2),
                      textcoords="offset points", ha="left", va="bottom", color=MUTED, fontsize=9)

    if slo:
        for ax in (ax_h, ax_t):
            ax.axvline(slo, color=MUTED, linewidth=1, linestyle="--", zorder=0)
        ax_h.annotate(f"SLO {fmt(slo)}", (slo, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                      textcoords="offset points", color=MUTED, fontsize=9)

    ax_h.set_title("Distribution", loc="left", color=INK, fontsize=11)
    ax_h.set_ylabel("% of requests per bin", color=MUTED)
    ax_t.set_title("Tail: share of requests slower than x", loc="left", color=INK, fontsize=11)
    ax_t.set_ylabel("% of requests slower", color=MUTED)
    ax_t.set_yscale("log")
    ax_t.set_ylim(bottom=min(0.05, 50 / max(len(v) for v in data.values())), top=150)
    ax_t.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}%"))

    for ax in (ax_h, ax_t):
        ax.set_xscale("log")
        ax.set_xlabel("latency (ms, log scale)", color=MUTED)
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
    ax_h.legend(frameon=False, labelcolor=INK)

    fig.savefig(out, dpi=150, facecolor="white")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", help="CSV written by the bench with -samples")
    ap.add_argument("--model", required=True, choices=["llm", "embed", "rerank"])
    ap.add_argument("--tier", required=True, choices=["premium", "free"])
    ap.add_argument("--metric", choices=["ttft", "itl", "e2e"], help="default: ttft for llm, e2e otherwise")
    ap.add_argument("--policies", help="comma-separated policies to include (default: all in the file)")
    ap.add_argument("--slo", type=float, help="draw a target line at this many ms")
    ap.add_argument("--out", help="PNG path (default: next to the CSV)")
    args = ap.parse_args()

    metric = args.metric or ("ttft" if args.model == "llm" else "e2e")
    data = load(args.samples, args.model, args.tier, metric)
    if args.policies:
        keep = [p.strip() for p in args.policies.split(",")]
        data = {p: data[p] for p in keep if p in data}
    data = {p: v for p, v in data.items() if len(v)}
    if not data:
        sys.exit(f"no {metric} samples for {args.model} {args.tier} in {args.samples}")

    names = {"ttft": "time to first token", "itl": "inter-token latency", "e2e": "end-to-end latency"}
    title = f"{args.model} {args.tier}: {names[metric]}"
    print(title)
    table(data)

    out = args.out or str(Path(args.samples).with_name(f"{Path(args.samples).stem}-{args.model}-{args.tier}-{metric}.png"))
    plot(data, title, out, args.slo)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
