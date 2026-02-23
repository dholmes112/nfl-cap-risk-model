from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"


TEAM_ABBR_TO_SLUG = {
    "ARI": "arizona-cardinals",
    "ATL": "atlanta-falcons",
    "BAL": "baltimore-ravens",
    "BUF": "buffalo-bills",
    "CAR": "carolina-panthers",
    "CHI": "chicago-bears",
    "CIN": "cincinnati-bengals",
    "CLE": "cleveland-browns",
    "DAL": "dallas-cowboys",
    "DEN": "denver-broncos",
    "DET": "detroit-lions",
    "GB": "green-bay-packers",
    "HOU": "houston-texans",
    "IND": "indianapolis-colts",
    "JAX": "jacksonville-jaguars",
    "KC": "kansas-city-chiefs",
    "LV": "las-vegas-raiders",
    "LAC": "los-angeles-chargers",
    "LAR": "los-angeles-rams",
    "MIA": "miami-dolphins",
    "MIN": "minnesota-vikings",
    "NE": "new-england-patriots",
    "NO": "new-orleans-saints",
    "NYG": "new-york-giants",
    "NYJ": "new-york-jets",
    "PHI": "philadelphia-eagles",
    "PIT": "pittsburgh-steelers",
    "SEA": "seattle-seahawks",
    "SF": "san-francisco-49ers",
    "TB": "tampa-bay-buccaneers",
    "TEN": "tennessee-titans",
    "WAS": "washington-commanders",
    "WSH": "washington-commanders",
    "OAK": "las-vegas-raiders",
    "SD": "los-angeles-chargers",
    "STL": "los-angeles-rams",
    "LA": "los-angeles-rams",
}


def normalize_team_slug(value: str) -> str:
    return value.strip().lower().replace(" ", "-").replace("_", "-")


def build_outcomes(min_year: int, max_year: int) -> pd.DataFrame:
    games = pd.read_csv(GAMES_URL, low_memory=False)
    game_type_col = "week_type" if "week_type" in games.columns else "game_type"
    needed_cols = ["season", game_type_col, "home_team", "away_team", "home_score", "away_score"]
    games = games[needed_cols].copy()
    games = games.rename(columns={game_type_col: "game_type"})
    games = games[(games["season"] >= min_year) & (games["season"] <= max_year)]

    reg = games[games["game_type"] == "REG"].copy()
    reg["home_win"] = (reg["home_score"] > reg["away_score"]).astype(int)
    reg["away_win"] = (reg["away_score"] > reg["home_score"]).astype(int)
    reg["tie"] = (reg["away_score"] == reg["home_score"]).astype(int)

    home = reg[["season", "home_team", "home_win", "tie"]].rename(
        columns={"home_team": "abbr", "home_win": "wins"}
    )
    away = reg[["season", "away_team", "away_win", "tie"]].rename(
        columns={"away_team": "abbr", "away_win": "wins"}
    )
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["wins"] = team_games["wins"] + 0.5 * team_games["tie"]

    wins = (
        team_games.groupby(["season", "abbr"], as_index=False)["wins"]
        .sum()
        .rename(columns={"season": "Year", "wins": "Wins"})
    )

    playoff_types = {"WC", "DIV", "CON", "SB", "POST"}
    post = games[games["game_type"].isin(playoff_types)].copy()
    post_home = post[["season", "home_team"]].rename(columns={"season": "Year", "home_team": "abbr"})
    post_away = post[["season", "away_team"]].rename(columns={"season": "Year", "away_team": "abbr"})
    post_teams = pd.concat([post_home, post_away], ignore_index=True).drop_duplicates()
    post_teams["Playoff_Appearance"] = 1

    outcomes = wins.merge(
        post_teams[["Year", "abbr", "Playoff_Appearance"]],
        on=["Year", "abbr"],
        how="left",
    )
    outcomes["Playoff_Appearance"] = outcomes["Playoff_Appearance"].fillna(0).astype(int)
    outcomes["Team"] = outcomes["abbr"].map(TEAM_ABBR_TO_SLUG)
    outcomes = outcomes.dropna(subset=["Team"]).copy()
    outcomes["Team"] = outcomes["Team"].map(normalize_team_slug)

    return outcomes[["Team", "Year", "Wins", "Playoff_Appearance"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Add wins and playoff appearance outcomes to team-season panel.")
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=Path("data/team_season_panel_2018_2025.csv"),
    )
    parser.add_argument("--min-year", type=int, default=2018)
    parser.add_argument("--max-year", type=int, default=2025)
    args = parser.parse_args()

    panel = pd.read_csv(args.panel_path)
    panel["Team"] = panel["Team"].astype(str).map(normalize_team_slug)
    panel["Year"] = pd.to_numeric(panel["Year"], errors="coerce").astype(int)

    outcomes = build_outcomes(args.min_year, args.max_year)
    merged = panel.drop(columns=["Wins", "Playoff_Appearance"], errors="ignore").merge(
        outcomes,
        on=["Team", "Year"],
        how="left",
    )

    merged.to_csv(args.panel_path, index=False)
    print(f"Updated outcomes in {args.panel_path}")
    print(f"- rows: {len(merged)}")
    print(f"- missing Wins: {int(merged['Wins'].isna().sum())}")
    print(f"- missing Playoff_Appearance: {int(merged['Playoff_Appearance'].isna().sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
