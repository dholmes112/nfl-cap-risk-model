from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from clean_data import clean_cap_data


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


def normalize_team(value: str) -> str:
    team = value.strip().lower().replace(" ", "-").replace("_", "-")
    aliases = {
        "cardinals": "arizona-cardinals",
        "ravens": "baltimore-ravens",
        "eagles": "philadelphia-eagles",
    }
    return aliases.get(team, team)


def extract_metric(summary: pd.DataFrame, label_contains: str) -> float:
    row = summary[summary["label"].str.contains(label_contains, na=False)].head(1)
    if row.empty:
        return float("nan")
    return parse_money(row.iloc[0]["value"])


def parse_team_from_filename(path: Path) -> str:
    stem = path.stem
    if "_overview_" in stem:
        return stem.split("_overview_")[0]
    return stem


OFFENSE_POSITIONS = {
    "QB", "RB", "FB", "HB", "TB", "WR", "TE",
    "LT", "LG", "C", "RG", "RT", "OL", "OT", "OG", "OC", "G",
}
DEFENSE_POSITIONS = {
    "DE", "DT", "NT", "DL", "EDGE",
    "OLB", "ILB", "LB", "MLB",
    "CB", "S", "FS", "SS", "DB",
}


def load_contracts_for_raw(raw_path: Path, processed_dir: Path) -> pd.DataFrame | None:
    processed_path = processed_dir / f"{raw_path.stem}.csv"
    if processed_path.exists():
        return pd.read_csv(processed_path)

    tables = pd.read_html(raw_path)
    if tables:
        return clean_cap_data(tables[0])
    return None


def compute_contract_metrics(contracts: pd.DataFrame, total_cap: float) -> dict[str, float]:
    if contracts is None or contracts.empty:
        return {
            "qb_cap_hit": float("nan"),
            "qb_cap_pct": float("nan"),
            "offensive_spend": float("nan"),
            "defensive_spend": float("nan"),
            "avg_roster_age": float("nan"),
        }

    contracts = contracts.copy()
    if "Cap Hit" in contracts.columns:
        cap_hit = pd.to_numeric(contracts["Cap Hit"], errors="coerce")
    else:
        cap_hit = pd.Series([float("nan")] * len(contracts))

    pos = contracts["Pos"].astype(str).str.upper() if "Pos" in contracts.columns else pd.Series([""] * len(contracts))
    qb_mask = pos.eq("QB")
    qb_cap_hit = cap_hit[qb_mask].sum(min_count=1)

    off_mask = pos.isin(OFFENSE_POSITIONS)
    def_mask = pos.isin(DEFENSE_POSITIONS)

    offensive_spend = cap_hit[off_mask].sum(min_count=1)
    defensive_spend = cap_hit[def_mask].sum(min_count=1)

    if "Age" in contracts.columns:
        age = pd.to_numeric(contracts["Age"], errors="coerce")
        avg_roster_age = age.mean()
    else:
        avg_roster_age = float("nan")

    if not total_cap or pd.isna(total_cap) or pd.isna(qb_cap_hit):
        qb_cap_pct = float("nan")
    else:
        qb_cap_pct = float(qb_cap_hit) / float(total_cap)

    return {
        "qb_cap_hit": qb_cap_hit,
        "qb_cap_pct": qb_cap_pct,
        "offensive_spend": offensive_spend,
        "defensive_spend": defensive_spend,
        "avg_roster_age": avg_roster_age,
    }


def normalize_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError("summary table must have at least 2 columns")

    normalized = df.copy().iloc[:, :3].copy()
    while normalized.shape[1] < 3:
        normalized[normalized.shape[1]] = pd.NA
    normalized.columns = ["label", "value", "rank"]
    normalized["label"] = normalized["label"].astype(str).str.strip()
    return normalized


def find_summary_table(tables: list[pd.DataFrame], preferred_index: int | None) -> pd.DataFrame:
    if preferred_index is not None and preferred_index < len(tables):
        candidate = normalize_summary_frame(tables[preferred_index])
        if candidate["label"].str.contains("NFL Salary Cap", na=False).any():
            return candidate

    for table in tables:
        try:
            candidate = normalize_summary_frame(table)
        except Exception:
            continue
        if candidate["label"].str.contains("NFL Salary Cap", na=False).any():
            return candidate

    raise ValueError("Could not locate summary table containing 'NFL Salary Cap'")


def parse_row_from_html(
    path: Path,
    team_override: str | None,
    processed_dir: Path,
    summary_table_index: int | None = None,
) -> dict[str, object]:
    tables = pd.read_html(path)
    summary = find_summary_table(tables, summary_table_index)

    salary_row = summary[summary["label"].str.contains("NFL Salary Cap", na=False)].head(1)
    if salary_row.empty:
        raise ValueError(f"Could not find NFL Salary Cap row in {path.name}")

    year_token = str(salary_row.iloc[0]["label"]).split()[0]
    year = int(year_token)

    team = team_override if team_override else parse_team_from_filename(path)
    team = normalize_team(team)

    total_cap = parse_money(salary_row.iloc[0]["value"])
    active_cap = extract_metric(summary, "Active Roster")
    dead_cap = extract_metric(summary, "Dead Money")
    contracts = load_contracts_for_raw(path, processed_dir)
    contract_metrics = compute_contract_metrics(contracts, total_cap)

    return {
        "Team": team,
        "Year": year,
        "Salary_Cap": total_cap,
        "Active_Cap_Spend": active_cap,
        "Dead_Cap": dead_cap,
        "Dead_Cap_%": (dead_cap / total_cap) if (not pd.isna(dead_cap) and not pd.isna(total_cap) and total_cap != 0) else float("nan"),
        "QB_Cap_Hit": contract_metrics["qb_cap_hit"],
        "QB_Cap_%": contract_metrics["qb_cap_pct"],
        "Offensive_Spend": contract_metrics["offensive_spend"],
        "Defensive_Spend": contract_metrics["defensive_spend"],
        "Avg_Roster_Age": contract_metrics["avg_roster_age"],
    }


def load_outcomes(outcomes_path: Path) -> pd.DataFrame:
    outcomes = pd.read_csv(outcomes_path)
    required = {"Team", "Year", "Wins", "Playoffs"}
    missing = required - set(outcomes.columns)
    if missing:
        raise ValueError(f"Outcomes file missing required columns: {sorted(missing)}")

    outcomes = outcomes.copy()
    outcomes["Team"] = outcomes["Team"].astype(str).map(normalize_team)
    outcomes["Year"] = pd.to_numeric(outcomes["Year"], errors="coerce").astype("Int64")
    return outcomes


def build_panel(
    raw_dir: Path,
    processed_dir: Path,
    output_path: Path,
    team: str | None,
    summary_table_index: int | None,
    outcomes_path: Path | None,
    min_year: int | None,
    max_year: int | None,
) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*_overview_*.html"))
    if not files:
        raise ValueError(f"No overview html files found in {raw_dir}")

    rows: list[dict[str, object]] = []
    for path in files:
        rows.append(parse_row_from_html(path, team, processed_dir, summary_table_index))

    panel = pd.DataFrame(rows).sort_values(["Team", "Year"]).reset_index(drop=True)
    panel = panel.drop_duplicates(subset=["Team", "Year"], keep="first").reset_index(drop=True)

    if min_year is not None:
        panel = panel[panel["Year"] >= min_year]
    if max_year is not None:
        panel = panel[panel["Year"] <= max_year]
    panel = panel.reset_index(drop=True)

    if outcomes_path:
        outcomes = load_outcomes(outcomes_path)
        if "Playoffs" in outcomes.columns and "Playoff_Appearance" not in outcomes.columns:
            outcomes = outcomes.rename(columns={"Playoffs": "Playoff_Appearance"})
        panel = panel.merge(outcomes, on=["Team", "Year"], how="left")
    else:
        panel["Wins"] = pd.NA
        panel["Playoff_Appearance"] = pd.NA

    ordered_cols = [
        "Team",
        "Year",
        "Salary_Cap",
        "Active_Cap_Spend",
        "Dead_Cap",
        "Dead_Cap_%",
        "QB_Cap_Hit",
        "QB_Cap_%",
        "Offensive_Spend",
        "Defensive_Spend",
        "Avg_Roster_Age",
        "Wins",
        "Playoff_Appearance",
    ]
    panel = panel[ordered_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build team-season panel for Phase 1 from raw Spotrac overview snapshots."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-path", type=Path, default=Path("data/team_season_panel.csv"))
    parser.add_argument("--team", type=str, default=None, help="Override team slug for all rows.")
    parser.add_argument(
        "--summary-table-index",
        type=int,
        default=None,
        help="Optional preferred table index containing team cap summary.",
    )
    parser.add_argument(
        "--outcomes-path",
        type=Path,
        default=None,
        help="Optional CSV with Team,Year,Wins,Playoffs to merge.",
    )
    parser.add_argument("--min-year", type=int, default=None)
    parser.add_argument("--max-year", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    panel = build_panel(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        output_path=args.output_path,
        team=args.team,
        summary_table_index=args.summary_table_index,
        outcomes_path=args.outcomes_path,
        min_year=args.min_year,
        max_year=args.max_year,
    )

    print("Built team-season panel")
    print(f"- rows: {len(panel)}")
    print(f"- output: {args.output_path}")
    print(panel.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
