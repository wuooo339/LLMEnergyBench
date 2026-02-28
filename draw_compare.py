#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


def parse_delta(value):
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if not text or text.lower() == "n/a":
        return 0.0
    return float(text.replace("%", "").replace("+", ""))


def build_plot(df: pd.DataFrame, output_path: Path) -> None:
    metric_groups = {
        "Higher is Better (Throughput)": [
            "request_throughput(req/s)",
            "output_throughput(tok/s)",
            "total_token_throughput(tok/s)",
        ],
        "Lower is Better (Latency)": [
            "mean_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "p99_tpot_ms",
            "mean_e2e_latency_ms",
            "p99_e2e_latency_ms",
        ],
        "Lower is Better (Energy)": [
            "energy.total_energy(J)",
            "energy.energy_per_request(J)",
            "energy.energy_per_token(J)",
        ],
    }

    order = [m for metrics in metric_groups.values() for m in metrics]
    map_df = pd.DataFrame(
        [(cat, metric) for cat, metrics in metric_groups.items() for metric in metrics],
        columns=["Category", "metric"],
    )

    key = df[df["metric"].isin(order)].copy()
    key = key.merge(map_df, on="metric", how="inner")
    key["delta_pct"] = key["delta(on-off)"].apply(parse_delta)

    def is_good(row):
        if "Lower is Better" in row["Category"]:
            return row["delta_pct"] < 0
        return row["delta_pct"] > 0

    key["is_good"] = key.apply(is_good, axis=1)
    key["color"] = key["is_good"].map({True: "#2ecc71", False: "#e74c3c"})
    key["metric"] = pd.Categorical(key["metric"], categories=order, ordered=True)
    key = key.sort_values("metric")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 14),
        gridspec_kw={"height_ratios": [1.2, 2.2, 1.2]},
    )
    fig.suptitle("Page Eviction A/B Analysis (ON vs OFF)", fontsize=18, fontweight="bold", y=0.98)

    for ax, category in zip(axes, metric_groups.keys()):
        sub = key[key["Category"] == category]
        if sub.empty:
            ax.set_visible(False)
            continue

        bars = ax.barh(
            sub["metric"],
            sub["delta_pct"],
            color=sub["color"],
            edgecolor="black",
            linewidth=0.8,
        )

        max_val = float(sub["delta_pct"].max())
        min_val = float(sub["delta_pct"].min())
        margin = max(abs(max_val), abs(min_val)) * 0.25 + 5.0
        ax.set_xlim(min_val - margin, max_val + margin)

        for bar in bars:
            width = bar.get_width()
            label_x = width + (margin * 0.05 if width >= 0 else -margin * 0.05)
            align = "left" if width >= 0 else "right"
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{width:+.2f}%",
                va="center",
                ha=align,
                fontweight="bold",
                fontsize=11,
            )

        ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
        ax.set_title(category, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Percentage Change (%)", fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=11)
        ax.invert_yaxis()

    legend_elements = [
        Patch(facecolor="#2ecc71", label="Improvement"),
        Patch(facecolor="#e74c3c", label="Regression"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Draw Page Eviction A/B comparison plot.")
    parser.add_argument("csv", type=Path, help="Path to page_eviction_ab_summary_*.csv")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    out = args.output if args.output is not None else args.csv.with_suffix(".png")
    df = pd.read_csv(args.csv)
    build_plot(df, out)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
