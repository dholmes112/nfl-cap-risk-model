from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def run_analysis(panel_path: Path) -> tuple[float, pd.DataFrame]:
    df = pd.read_csv(panel_path)

    required = ["Dead_Cap_%", "Wins", "Playoff_Appearance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    base = df[["Dead_Cap_%", "Wins", "Playoff_Appearance"]].dropna().copy()
    corr = float(base[["Dead_Cap_%", "Wins"]].corr().iloc[0, 1])

    base["dead_cap_quartile"] = pd.qcut(
        base["Dead_Cap_%"],
        4,
        labels=["Q1_Lowest_DeadCap", "Q2", "Q3", "Q4_Highest_DeadCap"],
    )

    summary = (
        base.groupby("dead_cap_quartile", observed=True)
        .agg(
            avg_wins=("Wins", "mean"),
            playoff_rate=("Playoff_Appearance", "mean"),
            n=("Wins", "size"),
        )
        .reset_index()
    )

    return corr, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Question 1 analysis: Does dead cap hurt performance?"
    )
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=Path("data/team_season_panel_2018_2025.csv"),
        help="Path to team-season panel CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional path to write quartile summary CSV.",
    )
    args = parser.parse_args()

    corr, summary = run_analysis(args.panel_path)

    print("Question 1: Does Dead Cap Hurt Performance?")
    print(f"Panel: {args.panel_path}")
    print(f"Rows used: {int(summary['n'].sum())}")
    print(f"Correlation (Dead_Cap_% vs Wins): {corr:.4f}")
    print("\nQuartile summary:")
    print(summary.to_string(index=False))

    low = summary.iloc[0]
    high = summary.iloc[-1]

    print("\nTop-vs-bottom contrast:")
    print(f"Lowest Dead Cap quartile avg wins: {low['avg_wins']:.3f}")
    print(f"Highest Dead Cap quartile avg wins: {high['avg_wins']:.3f}")
    print(f"Difference (Low - High): {(low['avg_wins'] - high['avg_wins']):.3f}")
    print(f"Lowest Dead Cap playoff rate: {100 * low['playoff_rate']:.1f}%")
    print(f"Highest Dead Cap playoff rate: {100 * high['playoff_rate']:.1f}%")

    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
        print(f"\nSaved quartile summary to: {args.summary_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
