from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import pandas as pd

from clean_data import clean_cap_data


@dataclass
class ModelConfig:
    horizon_years: int = 3
    cap_growth_rate: float = 0.075
    contract_decay: float = 0.9
    dead_cap_decay: float = 0.65
    restructure_fraction: float = 0.2
    restructure_min_cap_hit: float = 8_000_000
    restructure_allocation: tuple[float, float] = (0.6, 0.4)


def default_contracts_path() -> Path:
    base_dir = Path(__file__).resolve().parent.parent
    processed = base_dir / "data" / "processed" / "cardinals_overview_2026.csv"
    if processed.exists():
        return processed
    return base_dir / "data" / "cardinals_page.html"


def default_context_path() -> Path:
    base_dir = Path(__file__).resolve().parent.parent
    raw = base_dir / "data" / "raw" / "cardinals_overview_2026.html"
    if raw.exists():
        return raw
    return base_dir / "data" / "cardinals_page.html"


def parse_money(value: object) -> float:
    if pd.isna(value):
        return float("nan")

    text = str(value).strip()
    if not text:
        return float("nan")

    text = text.replace("$", "").replace(",", "")
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"

    try:
        return float(text)
    except ValueError:
        return float("nan")


def load_contracts(contracts_path: Path, table_index: int = 0) -> pd.DataFrame:
    if contracts_path.suffix.lower() == ".csv":
        contracts = pd.read_csv(contracts_path)
    else:
        tables = pd.read_html(contracts_path)
        if table_index >= len(tables):
            raise ValueError(f"table_index={table_index} out of range (found {len(tables)} tables)")
        contracts = tables[table_index]

    contracts = clean_cap_data(contracts)
    contracts["Age"] = pd.to_numeric(contracts["Age"], errors="coerce")
    contracts["Free Agent  Year"] = pd.to_numeric(contracts["Free Agent  Year"], errors="coerce")

    # Use positive dead cap for exposure modeling.
    contracts["Dead Cap"] = contracts["Dead Cap"].abs()

    required = ["Player (59)", "Age", "Cap Hit", "Dead Cap", "Cash Total", "Free Agent  Year"]
    missing = [c for c in required if c not in contracts.columns]
    if missing:
        raise ValueError(f"Missing required columns in contracts table: {missing}")

    return contracts


def load_team_cap_context(context_path: Path, summary_table_index: int = 3) -> dict[str, float | int]:
    tables = pd.read_html(context_path)
    if summary_table_index >= len(tables):
        raise ValueError(f"summary_table_index={summary_table_index} out of range (found {len(tables)} tables)")

    summary = tables[summary_table_index]
    summary = summary.rename(columns={0: "label", 1: "value", 2: "rank"})
    summary["label"] = summary["label"].astype(str).str.strip()

    salary_row = summary[summary["label"].str.contains("NFL Salary Cap", na=False)].head(1)
    adjusted_row = summary[summary["label"].str.contains("Adjusted Salary Cap", na=False)].head(1)
    space_row = summary[summary["label"].str.contains("Cap Space \(All\)", na=False)].head(1)

    if salary_row.empty:
        raise ValueError("Could not locate 'NFL Salary Cap' in team summary table")

    label = salary_row.iloc[0]["label"]
    year_token = str(label).split()[0]
    base_year = int(year_token)

    salary_cap = parse_money(salary_row.iloc[0]["value"])
    adjusted_cap = parse_money(adjusted_row.iloc[0]["value"]) if not adjusted_row.empty else salary_cap
    cap_space_all = parse_money(space_row.iloc[0]["value"]) if not space_row.empty else float("nan")

    return {
        "base_year": base_year,
        "salary_cap": salary_cap,
        "adjusted_cap": adjusted_cap,
        "cap_space_all": cap_space_all,
    }


def forecast_caps(base_year: int, adjusted_cap: float, growth_rate: float, horizon_years: int) -> dict[int, float]:
    return {
        base_year + i: adjusted_cap * ((1 + growth_rate) ** i)
        for i in range(horizon_years)
    }


def build_projection(
    contracts: pd.DataFrame,
    cap_forecast: dict[int, float],
    base_year: int,
    conservative: bool,
    config: ModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    restructure_rows: list[dict[str, float | str]] = []

    contracts = contracts.copy()
    contracts["years_left_from_base"] = (contracts["Free Agent  Year"] - base_year).clip(lower=0)

    eligible = contracts[(contracts["Cap Hit"] >= config.restructure_min_cap_hit) & (contracts["years_left_from_base"] >= 2)].copy()

    total_restructure = 0.0
    if not conservative and not eligible.empty:
        eligible["restructure_amount"] = eligible["Cap Hit"] * config.restructure_fraction
        total_restructure = eligible["restructure_amount"].sum()
        restructure_rows = eligible[["Player (59)", "Cap Hit", "restructure_amount"]].copy()
        restructure_rows = restructure_rows.sort_values("restructure_amount", ascending=False)
    else:
        restructure_rows = pd.DataFrame(columns=["Player (59)", "Cap Hit", "restructure_amount"])

    add_y1 = total_restructure * config.restructure_allocation[0]
    add_y2 = total_restructure * config.restructure_allocation[1]

    for year, team_cap in cap_forecast.items():
        t = year - base_year

        active = contracts[contracts["Free Agent  Year"] > year].copy()

        active_cap = (active["Cap Hit"] * (config.contract_decay**t)).sum()
        active_dead = (active["Dead Cap"] * (config.dead_cap_decay**t)).sum()

        if conservative:
            restructuring_now = 0.0
            restructure_carry = 0.0
            strategy_name = "conservative"
        else:
            if t == 0:
                restructuring_now = total_restructure
                restructure_carry = 0.0
            elif t == 1:
                restructuring_now = 0.0
                restructure_carry = add_y1
            else:
                restructuring_now = 0.0
                restructure_carry = add_y2
            strategy_name = "restructure-heavy"

        projected_commitments = active_cap - restructuring_now + restructure_carry
        projected_dead_cap = active_dead + (0.25 * restructure_carry)
        cap_space = team_cap - projected_commitments

        dead_cap_exposure = projected_dead_cap / projected_commitments if projected_commitments > 0 else float("nan")
        flexibility_ratio = cap_space / team_cap if team_cap > 0 else float("nan")
        obligation_pressure = projected_commitments / team_cap if team_cap > 0 else float("nan")
        restructure_ratio = restructuring_now / projected_commitments if projected_commitments > 0 else 0.0

        risk_score = (
            35 * min(max(dead_cap_exposure / 0.35, 0), 1)
            + 35 * min(max((0.18 - flexibility_ratio) / 0.18, 0), 1)
            + 20 * min(max((obligation_pressure - 0.72) / 0.28, 0), 1)
            + 10 * min(max(restructure_ratio / 0.12, 0), 1)
        )

        rows.append(
            {
                "strategy": strategy_name,
                "year": year,
                "team_cap": team_cap,
                "projected_commitments": projected_commitments,
                "projected_dead_cap": projected_dead_cap,
                "projected_cap_space": cap_space,
                "dead_cap_exposure": dead_cap_exposure,
                "obligation_pressure": obligation_pressure,
                "flexibility_ratio": flexibility_ratio,
                "risk_score_0_100": risk_score,
            }
        )

    return pd.DataFrame(rows), restructure_rows


def format_money(series: pd.Series) -> pd.Series:
    return series.map(lambda x: f"${x:,.0f}")


def render_results(context: dict[str, float | int], combined: pd.DataFrame, restructure_impact: pd.DataFrame) -> None:
    base_year = int(context["base_year"])

    print("\n3-Year Forward Financial Model")
    print(f"- Base year: {base_year}")
    print(f"- Adjusted cap baseline: ${context['adjusted_cap']:,.0f}")

    display = combined.copy()
    for money_col in ["team_cap", "projected_commitments", "projected_dead_cap", "projected_cap_space"]:
        display[money_col] = format_money(display[money_col])

    for pct_col in ["dead_cap_exposure", "obligation_pressure", "flexibility_ratio"]:
        display[pct_col] = (display[pct_col] * 100).map(lambda x: f"{x:.1f}%")

    display["risk_score_0_100"] = display["risk_score_0_100"].map(lambda x: f"{x:.1f}")

    print("\nTeam forecast by strategy")
    print(
        display[
            [
                "strategy",
                "year",
                "team_cap",
                "projected_commitments",
                "projected_dead_cap",
                "projected_cap_space",
                "dead_cap_exposure",
                "flexibility_ratio",
                "risk_score_0_100",
            ]
        ].to_string(index=False)
    )

    risk_summary = combined.groupby("strategy", as_index=False)["risk_score_0_100"].mean()
    risk_summary["risk_score_0_100"] = risk_summary["risk_score_0_100"].map(lambda x: f"{x:.1f}")

    print("\nAverage 3-year risk score")
    print(risk_summary.to_string(index=False))

    if not restructure_impact.empty:
        top = restructure_impact.copy()
        top["Cap Hit"] = format_money(top["Cap Hit"])
        top["restructure_amount"] = format_money(top["restructure_amount"])

        print("\nTop restructure candidates (year 1 cap converted)")
        print(top.head(10).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 3-year NFL cap risk and flexibility forecast.")
    parser.add_argument("--contracts-path", type=Path, default=default_contracts_path())
    parser.add_argument("--context-path", type=Path, default=default_context_path())
    parser.add_argument("--contracts-table-index", type=int, default=0, help="Only used when contracts-path is HTML.")
    parser.add_argument("--summary-table-index", type=int, default=3, help="Table index for cap summary in context HTML.")
    parser.add_argument("--horizon-years", type=int, default=3)
    parser.add_argument("--cap-growth-rate", type=float, default=0.075)
    parser.add_argument("--restructure-fraction", type=float, default=0.2)
    parser.add_argument("--restructure-min-cap-hit", type=float, default=8_000_000)

    args = parser.parse_args()

    config = ModelConfig(
        horizon_years=args.horizon_years,
        cap_growth_rate=args.cap_growth_rate,
        restructure_fraction=args.restructure_fraction,
        restructure_min_cap_hit=args.restructure_min_cap_hit,
    )

    contracts = load_contracts(args.contracts_path, args.contracts_table_index)
    context = load_team_cap_context(args.context_path, args.summary_table_index)

    cap_forecast = forecast_caps(
        base_year=int(context["base_year"]),
        adjusted_cap=float(context["adjusted_cap"]),
        growth_rate=config.cap_growth_rate,
        horizon_years=config.horizon_years,
    )

    conservative_df, _ = build_projection(
        contracts=contracts,
        cap_forecast=cap_forecast,
        base_year=int(context["base_year"]),
        conservative=True,
        config=config,
    )

    restructure_df, restructure_impact = build_projection(
        contracts=contracts,
        cap_forecast=cap_forecast,
        base_year=int(context["base_year"]),
        conservative=False,
        config=config,
    )

    combined = pd.concat([conservative_df, restructure_df], ignore_index=True)
    render_results(context, combined, restructure_impact)


if __name__ == "__main__":
    main()
