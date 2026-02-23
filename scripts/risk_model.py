from pathlib import Path
import argparse

import pandas as pd

from clean_data import clean_cap_data


def default_data_path() -> Path:
    base_dir = Path(__file__).resolve().parent.parent
    processed = base_dir / "data" / "processed" / "cardinals_overview_2026.csv"
    if processed.exists():
        return processed
    return base_dir / "data" / "cardinals_page.html"


def load_contracts(data_path: Path, table_index: int = 0) -> pd.DataFrame:
    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    else:
        tables = pd.read_html(data_path)
        if table_index >= len(tables):
            raise ValueError(f"table_index={table_index} out of range (found {len(tables)} tables)")
        df = tables[table_index]

    df = clean_cap_data(df)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    return df


def clean_player_display(value: object) -> str:
    text = str(value).strip()
    # Spotrac often formats as "Last  First Last"; keep the right-side full name.
    if "  " in text:
        return text.split("  ", 1)[1].strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute contract-level and team-level cap risk.")
    parser.add_argument("--data-path", type=Path, default=default_data_path())
    parser.add_argument("--table-index", type=int, default=0, help="Only used when data-path is an HTML file.")
    args = parser.parse_args()

    df = load_contracts(args.data_path, args.table_index)

    # Dead Cap Risk (contract rigidity)
    df["dead_cap_ratio"] = df["Dead Cap"].abs() / df["Cap Hit"]

    # Cash Exposure Risk
    df["cash_ratio"] = df["Cash Total"] / df["Cap Hit"]

    # Age Risk (starts at age 30)
    df["age_risk"] = ((df["Age"] - 29).clip(lower=0)) * 0.05

    # Composite Risk Score (weighted model)
    df["risk_score"] = (
        df["dead_cap_ratio"] * 0.5
        + df["age_risk"] * 0.3
        + df["cash_ratio"] * 0.2
    )

    # Remove infinite / NaN values
    df = df.replace([float("inf"), -float("inf")], pd.NA)
    df = df.dropna(subset=["risk_score"])

    # Sort by highest risk
    df = df.sort_values(by="risk_score", ascending=False)
    df["Player (59)"] = df["Player (59)"].map(clean_player_display)

    print("\nTop 10 Highest Risk Contracts:\n")
    print(df[["Player (59)", "Cap Hit", "Dead Cap", "Age", "risk_score"]].head(10))

    total_dead_cap = df["Dead Cap"].abs().sum()
    total_cap_hit = df["Cap Hit"].sum()
    team_dead_cap_ratio = total_dead_cap / total_cap_hit

    print("\nTeam Dead Cap Exposure Ratio:", round(team_dead_cap_ratio, 3))


if __name__ == "__main__":
    main()
