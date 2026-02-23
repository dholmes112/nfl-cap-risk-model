from pathlib import Path
import argparse
import sys

import pandas as pd

from clean_data import clean_cap_data


def default_data_path() -> Path:
    base_dir = Path(__file__).resolve().parent.parent
    processed = base_dir / "data" / "processed" / "cardinals_overview_2026.csv"
    if processed.exists():
        return processed
    return base_dir / "data" / "cardinals_page.html"


def load_table(data_path: Path, table_index: int = 0) -> tuple[pd.DataFrame, int]:
    if data_path.suffix.lower() == ".csv":
        return pd.read_csv(data_path), 1

    tables = pd.read_html(data_path)
    if not tables:
        raise ValueError("no tables found in HTML")
    if table_index >= len(tables):
        raise ValueError(f"table_index={table_index} out of range (found {len(tables)} tables)")
    return tables[table_index].copy(), len(tables)


def validate_input(data_path: Path, table_index: int = 0) -> int:
    issues = []

    if not data_path.exists():
        print(f"FAIL: file not found: {data_path}")
        return 1

    try:
        raw_df, tables_found = load_table(data_path, table_index)
    except Exception as exc:
        print(f"FAIL: unable to parse input data: {exc}")
        return 1

    required_cols = ["Age", "Cap Hit", "Dead Cap", "Cash Total"]
    missing = [col for col in required_cols if col not in raw_df.columns]
    if missing:
        print(f"FAIL: missing required columns: {missing}")
        return 1

    df = clean_cap_data(raw_df)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    if len(df) == 0:
        issues.append("table has zero rows")

    if (df["Cap Hit"] <= 0).any():
        issues.append("found Cap Hit <= 0; risk ratios would divide by zero or be invalid")

    # Reproduce risk-model columns and make sure they are finite
    df["dead_cap_ratio"] = df["Dead Cap"].abs() / df["Cap Hit"]
    df["cash_ratio"] = df["Cash Total"] / df["Cap Hit"]
    df["age_risk"] = ((df["Age"] - 29).clip(lower=0)) * 0.05
    df["risk_score"] = (
        df["dead_cap_ratio"] * 0.5
        + df["age_risk"] * 0.3
        + df["cash_ratio"] * 0.2
    )

    non_finite = ~pd.Series(pd.to_numeric(df["risk_score"], errors="coerce")).notna()

    cleaned = df.replace([float("inf"), -float("inf")], pd.NA).dropna(subset=["risk_score"])
    if len(cleaned) == 0:
        issues.append("all rows dropped after cleaning risk_score")

    print("Validation summary")
    print(f"- file: {data_path}")
    print(f"- tables_found: {tables_found}")
    print(f"- selected_table_shape: {raw_df.shape[0]} rows x {raw_df.shape[1]} cols")
    print(f"- rows_after_risk_cleaning: {len(cleaned)}")
    print(f"- rows_with_nan_risk_preclean: {int(non_finite.sum())}")
    print(f"- null_dead_cap: {int(df['Dead Cap'].isna().sum())}")
    print(f"- null_cash_total: {int(df['Cash Total'].isna().sum())}")

    if issues:
        print("\nFAIL")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("\nPASS: input is usable for risk assessment")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate contract input (CSV or HTML) before running risk model.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=default_data_path(),
        help="Path to processed CSV or saved Spotrac HTML page",
    )
    parser.add_argument(
        "--table-index",
        type=int,
        default=0,
        help="Which parsed HTML table to validate (ignored for CSV)",
    )

    args = parser.parse_args()
    return validate_input(args.data_path, args.table_index)


if __name__ == "__main__":
    sys.exit(main())



