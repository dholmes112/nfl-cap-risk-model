from pathlib import Path
import argparse

import pandas as pd

from clean_data import clean_cap_data


def base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def processed_dir() -> Path:
    return base_dir() / "data" / "processed"


def parse_team_year_from_filename(path: Path) -> tuple[str, int] | None:
    stem = path.stem
    if "_overview_" not in stem:
        return None
    team, year_text = stem.rsplit("_overview_", 1)
    if not year_text.isdigit():
        return None
    return team, int(year_text)


def available_team_years() -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for csv_path in processed_dir().glob("*_overview_*.csv"):
        parsed = parse_team_year_from_filename(csv_path)
        if not parsed:
            continue
        team, year = parsed
        out.setdefault(team, set()).add(year)
    return out


def default_data_path(team: str = "arizona-cardinals", year: int = 2026) -> Path:
    candidate = processed_dir() / f"{team}_overview_{year}.csv"
    if candidate.exists():
        return candidate
    legacy = processed_dir() / f"cardinals_overview_{year}.csv"
    if team == "arizona-cardinals" and legacy.exists():
        return legacy
    return base_dir() / "data" / "cardinals_page.html"


def resolve_data_path(team: str, year: int, data_path: Path | None) -> Path:
    if data_path is not None:
        return data_path

    candidate = processed_dir() / f"{team}_overview_{year}.csv"
    if candidate.exists():
        return candidate

    if team == "arizona-cardinals":
        legacy = processed_dir() / f"cardinals_overview_{year}.csv"
        if legacy.exists():
            return legacy

    team_years = available_team_years()
    if team in team_years:
        years = ", ".join(str(y) for y in sorted(team_years[team]))
        raise FileNotFoundError(
            f"No processed file for team='{team}' and year={year}. "
            f"Available years for this team: {years}"
        )

    sample_teams = ", ".join(sorted(team_years.keys())[:10])
    raise FileNotFoundError(
        f"Unknown team slug '{team}'. "
        f"Use --list-teams to see all options. Example teams: {sample_teams}"
    )


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
    parser.add_argument("team_pos", nargs="?", help="Team slug (positional), e.g. buffalo-bills")
    parser.add_argument("year_pos", nargs="?", type=int, help="Season year (positional), e.g. 2024")
    parser.add_argument("--team", type=str, default=None, help="Team slug, e.g. buffalo-bills")
    parser.add_argument("--year", type=int, default=None, help="Season year, e.g. 2024")
    parser.add_argument("--list-teams", action="store_true", help="List team slugs found in data/processed")
    parser.add_argument(
        "--list-years",
        action="store_true",
        help="List available years for the selected --team and exit",
    )
    parser.add_argument("--data-path", type=Path, default=None, help="Optional explicit CSV/HTML path override")
    parser.add_argument("--table-index", type=int, default=0, help="Only used when data-path is an HTML file.")
    args = parser.parse_args()

    selected_team = args.team_pos or args.team or "arizona-cardinals"
    selected_year = args.year_pos or args.year or 2026

    team_years = available_team_years()
    if args.list_teams:
        for team in sorted(team_years):
            print(team)
        return

    if args.list_years:
        years = sorted(team_years.get(selected_team, set()))
        if not years:
            raise ValueError(f"No processed data found for team '{selected_team}'")
        print(f"{selected_team}: {', '.join(str(y) for y in years)}")
        return

    data_path = resolve_data_path(selected_team, selected_year, args.data_path)
    df = load_contracts(data_path, args.table_index)
    player_col = next((col for col in df.columns if col.startswith("Player")), "Player (59)")

    # Dead Cap Risk (contract rigidity)
    df["dead_cap_ratio"] = df["Dead Cap"].abs() / df["Cap Hit"]

    # Cash Exposure Risk
    df["cash_ratio"] = df["Cash Total"] / df["Cap Hit"]

    # Age Risk (starts at age 30)
    df["age_risk"] = ((df["Age"] - 29).clip(lower=0)) * 0.05

    # Composite Risk Score (weighted model)
    df["risk_score_raw"] = (
        df["dead_cap_ratio"] * 0.5
        + df["age_risk"] * 0.3
        + df["cash_ratio"] * 0.2
    )

    # Remove infinite / NaN values
    df = df.replace([float("inf"), -float("inf")], pd.NA)
    df = df.dropna(subset=["risk_score_raw"])

    # Normalize to a 0-100 scale for easier interpretation.
    min_score = df["risk_score_raw"].min()
    max_score = df["risk_score_raw"].max()
    if max_score == min_score:
        df["risk_score"] = 100.0
    else:
        df["risk_score"] = ((df["risk_score_raw"] - min_score) / (max_score - min_score)) * 100
    df["risk_score"] = df["risk_score"].round(2)

    # Sort by highest risk
    df = df.sort_values(by="risk_score", ascending=False)
    df["Player"] = df[player_col].map(clean_player_display)

    print(f"\nTop 10 Highest Risk Contracts ({selected_team}, {selected_year}):\n")
    print(df[["Player", "Cap Hit", "Dead Cap", "Age", "risk_score"]].head(10))

    total_dead_cap = df["Dead Cap"].abs().sum()
    total_cap_hit = df["Cap Hit"].sum()
    team_dead_cap_ratio = total_dead_cap / total_cap_hit

    print("\nTeam Dead Cap Exposure Ratio:", round(team_dead_cap_ratio, 3))


if __name__ == "__main__":
    main()
