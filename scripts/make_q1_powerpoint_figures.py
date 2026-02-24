from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    q = df[["Dead_Cap_%", "Wins", "Playoff_Appearance"]].dropna().copy()
    q["dead_cap_quartile"] = pd.qcut(
        q["Dead_Cap_%"],
        4,
        labels=["Q1 Lowest", "Q2", "Q3", "Q4 Highest"],
    )

    out = (
        q.groupby("dead_cap_quartile", observed=True)
        .agg(
            avg_wins=("Wins", "mean"),
            playoff_rate=("Playoff_Appearance", "mean"),
            n=("Wins", "size"),
        )
        .reset_index()
    )
    return out


def save_wins_bar(plt, quartiles: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(quartiles["dead_cap_quartile"], quartiles["avg_wins"], color=["#2E86AB", "#6CA6C1", "#9BC4D8", "#D1495B"])

    ax.set_title("Average Wins by Dead Cap Quartile (2018-2025)")
    ax.set_xlabel("Dead Cap Quartile")
    ax.set_ylabel("Average Wins")
    ax.set_ylim(0, max(quartiles["avg_wins"]) * 1.25)

    for bar, val in zip(bars, quartiles["avg_wins"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_playoff_bar(plt, quartiles: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    rates = quartiles["playoff_rate"] * 100
    bars = ax.bar(quartiles["dead_cap_quartile"], rates, color=["#2E86AB", "#6CA6C1", "#9BC4D8", "#D1495B"])

    ax.set_title("Playoff Rate by Dead Cap Quartile (2018-2025)")
    ax.set_xlabel("Dead Cap Quartile")
    ax.set_ylabel("Playoff Rate (%)")
    ax.set_ylim(0, max(rates) * 1.25)

    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_scatter(plt, df: pd.DataFrame, out_path: Path) -> None:
    s = df[["Dead_Cap_%", "Wins"]].dropna().copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = s["Dead_Cap_%"].astype(float) * 100
    y = s["Wins"].astype(float)
    ax.scatter(x, y, alpha=0.6, color="#2E86AB", edgecolors="none")

    m, b = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    y_line = m * x_line + b
    ax.plot(x_line, y_line, color="#D1495B", linewidth=2)

    corr = s[["Dead_Cap_%", "Wins"]].corr().iloc[0, 1]

    ax.set_title("Dead Cap % vs Wins (All Teams, 2018-2025)")
    ax.set_xlabel("Dead Cap as % of Salary Cap")
    ax.set_ylabel("Wins")
    ax.text(0.02, 0.95, f"Correlation: {corr:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PowerPoint-ready figures for Question 1 dead-cap analysis.")
    parser.add_argument("--panel-path", type=Path, default=Path("data/team_season_panel_2018_2025.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/figures"))
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print("ERROR: matplotlib is not installed.")
        print("Install it with: python -m pip install matplotlib")
        raise SystemExit(1) from exc

    df = pd.read_csv(args.panel_path)
    required = ["Dead_Cap_%", "Wins", "Playoff_Appearance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    quartiles = build_quartiles(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    save_wins_bar(plt, quartiles, args.out_dir / "q1_avg_wins_by_dead_cap_quartile.png")
    save_playoff_bar(plt, quartiles, args.out_dir / "q1_playoff_rate_by_dead_cap_quartile.png")
    save_scatter(plt, df, args.out_dir / "q1_dead_cap_vs_wins_scatter.png")

    quartiles.to_csv(args.out_dir / "q1_dead_cap_quartile_summary.csv", index=False)

    print("Generated files:")
    print(f"- {args.out_dir / 'q1_avg_wins_by_dead_cap_quartile.png'}")
    print(f"- {args.out_dir / 'q1_playoff_rate_by_dead_cap_quartile.png'}")
    print(f"- {args.out_dir / 'q1_dead_cap_vs_wins_scatter.png'}")
    print(f"- {args.out_dir / 'q1_dead_cap_quartile_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
