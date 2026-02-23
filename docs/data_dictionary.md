# Data Dictionary

## Source: Cardinals overview table (`data/processed/cardinals_overview_YYYY.csv`)
- `Unnamed: 0`: Source row number/index from Spotrac table.
- `Player (59)`: Player display name (header count may vary by year).
- `Pos`: Player position.
- `Age`: Player age for the selected season.
- `Cap Hit`: Current-season salary cap charge for the player.
- `Cap Hit Pct  League Cap`: Player cap hit as a percent of the league salary cap.
- `Dead Cap`: Cap charge that can remain if the player is cut/traded (source-context dependent).
- `Cash Total`: Total cash paid to player in the selected season.
- `Free Agent  Year`: First projected free-agent year in source data.

## Key Cap Concepts
- `Dead Cap`: Cap charges that remain after separation (release/trade/void acceleration).
- `Void Years`: Contract accounting years used to spread proration; not true playing years.
- `Dead Cap vs Void Years`: void years are a structure mechanism, dead cap is the resulting cap impact.

## Derived Columns: `scripts/risk_model.py`
- `dead_cap_ratio = abs(Dead Cap) / Cap Hit`
- `cash_ratio = Cash Total / Cap Hit`
- `age_risk = max(Age - 29, 0) * 0.05`
- `risk_score = 0.5*dead_cap_ratio + 0.3*age_risk + 0.2*cash_ratio`

Formula meanings:
- `dead_cap_ratio`: Contract rigidity. Higher values mean it is more expensive to move on from the player relative to current cap hit.
- `cash_ratio`: Cash intensity. Higher values mean more real cash is being paid relative to cap accounting.
- `age_risk`: Age-based risk add-on. No penalty through age 29; then adds `0.05` per year above 29.
- `risk_score`: Weighted player risk index.  
  `50%` dead-cap rigidity + `30%` age effect + `20%` cash exposure.

Notes:
- Rows with invalid ratio math (`inf`, `-inf`, `NaN`) are cleaned before ranking.
- Current model captures financial rigidity risk, not on-field performance value.

## Validation Fields: `scripts/validate_input.py`
- `tables_found`: Number of parsed tables (HTML) or `1` for CSV input.
- `selected_table_shape`: Rows/columns in evaluated dataset.
- `rows_with_nan_risk_preclean`: Rows where risk score is NaN before cleaning.
- `rows_after_risk_cleaning`: Rows remaining after replacing `inf` and dropping NaN risk scores.
- `null_dead_cap`: Count of null dead cap values.
- `null_cash_total`: Count of null cash total values.

## Forward Model Outputs: `scripts/forward_financial_model.py`
- `strategy`: Forecast strategy (`conservative` or `restructure-heavy`).
- `year`: Forecast season year.
- `team_cap`: Projected team cap for that year.
- `projected_commitments`: Estimated cap commitments under strategy assumptions.
- `projected_dead_cap`: Estimated dead cap exposure under strategy assumptions.
- `projected_cap_space = team_cap - projected_commitments`
- `dead_cap_exposure = projected_dead_cap / projected_commitments`
- `obligation_pressure = projected_commitments / team_cap`
- `flexibility_ratio = projected_cap_space / team_cap`
- `risk_score_0_100`: Composite team financial risk score (higher = riskier).

Formula meanings:
- `projected_cap_space`: Remaining room after projected commitments.
- `dead_cap_exposure`: Share of commitments tied to dead-cap burden.
- `obligation_pressure`: How much of the cap is already consumed by commitments.
- `flexibility_ratio`: Percent of the cap still available to use.
- `risk_score_0_100`: Team-level financial risk index from 0 to 100; higher means less flexibility and more forward cap stress.

## Fetch Pipeline Outputs: `scripts/fetch_cardinals_data.py`
- Raw HTML snapshots: `data/raw/cardinals_overview_YYYY.html`
- Processed contracts CSV: `data/processed/cardinals_overview_YYYY.csv`

## Phase 1 Panel: `scripts/build_team_panel.py`
Output examples:
- `data/team_season_panel.csv`
- `data/team_season_panel_2018_2025.csv`

Current schema:
- `Team`: Team slug (example: `arizona-cardinals`).
- `Year`: Season year.
- `Salary_Cap`: League salary cap from team summary table.
- `Active_Cap_Spend`: Active roster cap commitments from team summary table.
- `Dead_Cap`: Dead money from team summary table.
- `Dead_Cap_% = Dead_Cap / Salary_Cap`
- `QB_Cap_Hit`: Sum of QB cap hits from contracts table.
- `QB_Cap_% = QB_Cap_Hit / Salary_Cap`
- `Offensive_Spend`: Sum of cap hit for offense positions.
- `Defensive_Spend`: Sum of cap hit for defense positions.
- `Avg_Roster_Age`: Mean age from contracts table.
- `Wins`: Optional outcomes merge field.
- `Playoff_Appearance`: Optional outcomes merge field (0/1).
