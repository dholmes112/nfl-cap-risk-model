from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing required column. Tried: {candidates}")


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _safe_rank_0_100(series: pd.Series) -> pd.Series:
    rank = series.rank(method="average", pct=True)
    return (rank * 100).clip(0, 100)


def _apply_direction(rank_0_100: pd.Series, low_is_risky: bool) -> pd.Series:
    if low_is_risky:
        return 100 - rank_0_100
    return rank_0_100


def _auc_binary(y_true: pd.Series, y_score: pd.Series) -> float:
    mask = y_true.notna() & y_score.notna()
    y = y_true[mask].astype(int).to_numpy()
    s = y_score[mask].to_numpy(dtype=float)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _fit_linear(train_x: pd.Series, train_y: pd.Series) -> tuple[float, float]:
    mask = train_x.notna() & train_y.notna()
    x = train_x[mask].to_numpy(dtype=float)
    y = train_y[mask].to_numpy(dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return y_mean, 0.0
    slope = ((x - x_mean) * (y - y_mean)).sum() / denom
    intercept = y_mean - slope * x_mean
    return float(intercept), float(slope)


def _predict_linear(x: pd.Series, intercept: float, slope: float) -> pd.Series:
    return intercept + slope * x


def _rmse(actual: pd.Series, pred: pd.Series) -> float:
    mask = actual.notna() & pred.notna()
    if not mask.any():
        return float("nan")
    err = (actual[mask] - pred[mask]).to_numpy(dtype=float)
    return float(np.sqrt((err**2).mean()))


def _mae(actual: pd.Series, pred: pd.Series) -> float:
    mask = actual.notna() & pred.notna()
    if not mask.any():
        return float("nan")
    err = (actual[mask] - pred[mask]).to_numpy(dtype=float)
    return float(np.abs(err).mean())


def _corr(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return float("nan")
    return float(pd.DataFrame({"x": x[mask], "y": y[mask]}).corr().iloc[0, 1])


def _weight_grid(step: float = 0.05) -> list[tuple[float, float, float]]:
    grid: list[tuple[float, float, float]] = []
    n = int(round(1 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            grid.append((i * step, j * step, k * step))
    return grid


def tune_weights(base_df: pd.DataFrame, train_max_year: int) -> dict[str, float | int | str]:
    tune_train_max = train_max_year - 2
    train = base_df[base_df["Year"] <= tune_train_max].copy()
    valid = base_df[(base_df["Year"] > tune_train_max) & (base_df["Year"] <= train_max_year)].copy()

    if train.empty or valid.empty:
        raise ValueError("Insufficient years to tune weights. Increase year range or adjust --train-max-year.")

    candidates: list[dict[str, float | int | str]] = []
    for w_dead, w_qb, w_spend in _weight_grid(step=0.05):
        if w_dead == 0 and w_qb == 0 and w_spend == 0:
            continue
        for qb_low_is_risky in (False, True):
            for spend_low_is_risky in (False, True):
                train_score = (
                    w_dead * train["dead_risk_0_100"]
                    + w_qb * _apply_direction(train["qb_risk_0_100"], qb_low_is_risky)
                    + w_spend * _apply_direction(train["spend_risk_0_100"], spend_low_is_risky)
                )
                valid_score = (
                    w_dead * valid["dead_risk_0_100"]
                    + w_qb * _apply_direction(valid["qb_risk_0_100"], qb_low_is_risky)
                    + w_spend * _apply_direction(valid["spend_risk_0_100"], spend_low_is_risky)
                )

                intercept, slope = _fit_linear(train_score, train["Wins"])
                valid_pred = _predict_linear(valid_score, intercept, slope)
                valid_rmse = _rmse(valid["Wins"], valid_pred)
                valid_auc = _auc_binary(valid["Playoff_Appearance"], -valid_score)
                valid_corr = _corr(valid_score, valid["Wins"])

                candidates.append(
                    {
                        "w_dead": w_dead,
                        "w_qb": w_qb,
                        "w_spend": w_spend,
                        "qb_low_is_risky": int(qb_low_is_risky),
                        "spend_low_is_risky": int(spend_low_is_risky),
                        "valid_rmse": valid_rmse,
                        "valid_auc": valid_auc,
                        "valid_corr": valid_corr,
                    }
                )

    results = pd.DataFrame(candidates).dropna(subset=["valid_rmse"]).copy()
    if results.empty:
        return {
            "w_dead": 0.5,
            "w_qb": 0.2,
            "w_spend": 0.3,
            "qb_low_is_risky": 0,
            "spend_low_is_risky": 0,
            "tune_train_max_year": int(tune_train_max),
            "tune_valid_start_year": int(tune_train_max + 1),
            "tune_valid_end_year": int(train_max_year),
            "tune_valid_rmse": float("nan"),
            "tune_valid_auc": float("nan"),
            "tune_valid_corr": float("nan"),
        }

    results["valid_auc_filled"] = results["valid_auc"].fillna(0.5)
    results["valid_corr_filled"] = results["valid_corr"].fillna(0.0)
    results["rmse_rank"] = results["valid_rmse"].rank(method="min", ascending=True)
    results["auc_rank"] = results["valid_auc_filled"].rank(method="min", ascending=False)
    results["corr_rank"] = results["valid_corr_filled"].rank(method="min", ascending=True)  # lower is better
    results["overall_rank"] = results["rmse_rank"] + results["auc_rank"] + results["corr_rank"]

    best = results.sort_values(["overall_rank", "valid_rmse", "valid_auc"], ascending=[True, True, False]).iloc[0]
    return {
        "w_dead": float(best["w_dead"]),
        "w_qb": float(best["w_qb"]),
        "w_spend": float(best["w_spend"]),
        "qb_low_is_risky": int(best["qb_low_is_risky"]),
        "spend_low_is_risky": int(best["spend_low_is_risky"]),
        "tune_train_max_year": int(tune_train_max),
        "tune_valid_start_year": int(tune_train_max + 1),
        "tune_valid_end_year": int(train_max_year),
        "tune_valid_rmse": float(best["valid_rmse"]),
        "tune_valid_auc": float(best["valid_auc"]),
        "tune_valid_corr": float(best["valid_corr"]),
    }


def build_validated_frame(panel: pd.DataFrame) -> pd.DataFrame:
    team_col = _pick_column(panel, ["Team"])
    year_col = _pick_column(panel, ["Year"])
    dead_cap_col = _pick_column(panel, ["Dead_Cap_%", "Dead Cap %"])
    wins_col = _pick_column(panel, ["Wins"])
    playoffs_col = _pick_column(panel, ["Playoff_Appearance", "Playoffs"])
    qb_cap_col = _pick_column(panel, ["QB_Cap_%", "QBCap%"])
    active_spend_col = _pick_column(panel, ["Active_Cap_Spend", "Active Cap"])
    salary_cap_col = _pick_column(panel, ["Salary_Cap", "Total Cap"])

    df = panel[
        [
            team_col,
            year_col,
            dead_cap_col,
            wins_col,
            playoffs_col,
            qb_cap_col,
            active_spend_col,
            salary_cap_col,
        ]
    ].copy()

    df.columns = [
        "Team",
        "Year",
        "Dead_Cap_Pct",
        "Wins",
        "Playoff_Appearance",
        "QB_Cap_Pct",
        "Active_Cap_Spend",
        "Salary_Cap",
    ]

    df = _to_numeric(
        df,
        [
            "Year",
            "Dead_Cap_Pct",
            "Wins",
            "Playoff_Appearance",
            "QB_Cap_Pct",
            "Active_Cap_Spend",
            "Salary_Cap",
        ],
    )
    df["Active_Spend_Pct"] = df["Active_Cap_Spend"] / df["Salary_Cap"]

    df = df.sort_values(["Team", "Year"]).reset_index(drop=True)
    df["Dead_Cap_Pct_YoY_Change"] = df.groupby("Team")["Dead_Cap_Pct"].diff()
    df["Restructure_Proxy"] = (
        (df["Dead_Cap_Pct_YoY_Change"] >= 0.03)
        & (df["Active_Spend_Pct"] >= 0.90)
    ).astype(int)

    # Team-season financial risk index in 0-100 using weighted percentile ranks.
    df["dead_risk_0_100"] = _safe_rank_0_100(df["Dead_Cap_Pct"])
    df["qb_risk_0_100"] = _safe_rank_0_100(df["QB_Cap_Pct"])
    df["spend_risk_0_100"] = _safe_rank_0_100(df["Active_Spend_Pct"])

    return df


def apply_risk_formula(df: pd.DataFrame, config: dict[str, float | int | str]) -> pd.DataFrame:
    out = df.copy()
    out["risk_score_0_100"] = (
        float(config["w_dead"]) * out["dead_risk_0_100"]
        + float(config["w_qb"]) * _apply_direction(out["qb_risk_0_100"], bool(config["qb_low_is_risky"]))
        + float(config["w_spend"]) * _apply_direction(out["spend_risk_0_100"], bool(config["spend_low_is_risky"]))
    ).round(2)
    return out


def summarize_validation(df: pd.DataFrame, train_max_year: int, config: dict[str, float | int | str]) -> pd.DataFrame:
    corr_dead_wins = df[["Dead_Cap_Pct", "Wins"]].corr().iloc[0, 1]
    corr_risk_wins = df[["risk_score_0_100", "Wins"]].corr().iloc[0, 1]

    restructure_summary = df.groupby("Restructure_Proxy", as_index=False).agg(
        Avg_Wins=("Wins", "mean"),
        Avg_Dead_Cap_Pct=("Dead_Cap_Pct", "mean"),
        Team_Seasons=("Team", "count"),
        Playoff_Rate=("Playoff_Appearance", "mean"),
    )

    train = df[df["Year"] <= train_max_year]
    test = df[df["Year"] > train_max_year]
    intercept, slope = _fit_linear(train["risk_score_0_100"], train["Wins"])

    test = test.copy()
    test["wins_pred"] = _predict_linear(test["risk_score_0_100"], intercept, slope)
    rmse = _rmse(test["Wins"], test["wins_pred"])
    mae = _mae(test["Wins"], test["wins_pred"])

    # Lower risk should imply better playoff odds, so invert score for AUC.
    auc = _auc_binary(test["Playoff_Appearance"], -test["risk_score_0_100"])

    summary_rows = [
        {"metric": "rows", "value": len(df)},
        {"metric": "min_year", "value": int(df["Year"].min())},
        {"metric": "max_year", "value": int(df["Year"].max())},
        {"metric": "corr_dead_cap_pct_vs_wins", "value": float(corr_dead_wins)},
        {"metric": "corr_risk_score_vs_wins", "value": float(corr_risk_wins)},
        {"metric": "train_max_year", "value": int(train_max_year)},
        {"metric": "train_rows", "value": len(train)},
        {"metric": "test_rows", "value": len(test)},
        {"metric": "wins_model_intercept", "value": float(intercept)},
        {"metric": "wins_model_slope", "value": float(slope)},
        {"metric": "wins_model_rmse_test", "value": float(rmse)},
        {"metric": "wins_model_mae_test", "value": float(mae)},
        {"metric": "playoff_auc_test", "value": float(auc)},
        {"metric": "weight_dead_cap", "value": float(config["w_dead"])},
        {"metric": "weight_qb_cap", "value": float(config["w_qb"])},
        {"metric": "weight_active_spend", "value": float(config["w_spend"])},
        {"metric": "qb_low_is_risky", "value": int(config["qb_low_is_risky"])},
        {"metric": "spend_low_is_risky", "value": int(config["spend_low_is_risky"])},
        {"metric": "tune_train_max_year", "value": int(config["tune_train_max_year"])},
        {"metric": "tune_valid_start_year", "value": int(config["tune_valid_start_year"])},
        {"metric": "tune_valid_end_year", "value": int(config["tune_valid_end_year"])},
        {"metric": "tune_valid_rmse", "value": float(config["tune_valid_rmse"])},
        {"metric": "tune_valid_auc", "value": float(config["tune_valid_auc"])},
        {"metric": "tune_valid_corr", "value": float(config["tune_valid_corr"])},
    ]

    for _, row in restructure_summary.iterrows():
        key = int(row["Restructure_Proxy"])
        summary_rows.extend(
            [
                {"metric": f"restructure_proxy_{key}_avg_wins", "value": float(row["Avg_Wins"])},
                {
                    "metric": f"restructure_proxy_{key}_avg_dead_cap_pct",
                    "value": float(row["Avg_Dead_Cap_Pct"]),
                },
                {
                    "metric": f"restructure_proxy_{key}_team_seasons",
                    "value": int(row["Team_Seasons"]),
                },
                {
                    "metric": f"restructure_proxy_{key}_playoff_rate",
                    "value": float(row["Playoff_Rate"]),
                },
            ]
        )

    return pd.DataFrame(summary_rows)


def make_deciles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["risk_decile"] = pd.qcut(out["risk_score_0_100"], q=10, labels=False, duplicates="drop") + 1
    deciles = out.groupby("risk_decile", as_index=False).agg(
        risk_score_avg=("risk_score_0_100", "mean"),
        dead_cap_pct_avg=("Dead_Cap_Pct", "mean"),
        wins_avg=("Wins", "mean"),
        playoff_rate=("Playoff_Appearance", "mean"),
        team_seasons=("Team", "count"),
    )
    return deciles


def summarize_validation_by_team(df: pd.DataFrame, train_max_year: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for team, tdf in df.groupby("Team"):
        tdf = tdf.sort_values("Year").copy()
        train = tdf[tdf["Year"] <= train_max_year]
        test = tdf[tdf["Year"] > train_max_year]

        intercept, slope = _fit_linear(train["risk_score_0_100"], train["Wins"])
        test_pred = _predict_linear(test["risk_score_0_100"], intercept, slope)
        rmse = _rmse(test["Wins"], test_pred)
        mae = _mae(test["Wins"], test_pred)
        auc = _auc_binary(test["Playoff_Appearance"], -test["risk_score_0_100"])

        rows.append(
            {
                "Team": team,
                "rows": int(len(tdf)),
                "min_year": int(tdf["Year"].min()),
                "max_year": int(tdf["Year"].max()),
                "corr_dead_cap_pct_vs_wins": _corr(tdf["Dead_Cap_Pct"], tdf["Wins"]),
                "corr_risk_score_vs_wins": _corr(tdf["risk_score_0_100"], tdf["Wins"]),
                "avg_wins": float(tdf["Wins"].mean()),
                "avg_dead_cap_pct": float(tdf["Dead_Cap_Pct"].mean()),
                "avg_risk_score_0_100": float(tdf["risk_score_0_100"].mean()),
                "playoff_rate": float(tdf["Playoff_Appearance"].mean()),
                "restructure_proxy_rate": float(tdf["Restructure_Proxy"].mean()),
                "wins_model_slope": float(slope),
                "wins_model_rmse_test": float(rmse),
                "wins_model_mae_test": float(mae),
                "playoff_auc_test": float(auc),
            }
        )
    return pd.DataFrame(rows).sort_values("Team").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate team-level cap risk model with historical panel data."
    )
    parser.add_argument(
        "team_pos",
        nargs="?",
        help="Optional team slug positional input, e.g. buffalo-bills",
    )
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=Path("data/team_season_panel_2018_2025.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/model_validation_summary.csv"),
    )
    parser.add_argument(
        "--decile-output",
        type=Path,
        default=Path("data/model_validation_by_decile.csv"),
    )
    parser.add_argument(
        "--scored-output",
        type=Path,
        default=Path("data/risk_vs_outcomes_scored.csv"),
    )
    parser.add_argument(
        "--team-summary-output",
        type=Path,
        default=Path("data/model_validation_summary_by_team.csv"),
    )
    parser.add_argument(
        "--train-max-year",
        type=int,
        default=2022,
        help="Train through this year, evaluate on later years.",
    )
    parser.add_argument(
        "--team",
        type=str,
        default=None,
        help="Optional team slug filter, e.g. buffalo-bills",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    panel = pd.read_csv(args.panel_path)
    selected_team = args.team_pos or args.team
    if selected_team:
        panel = panel[panel["Team"].astype(str).str.lower() == selected_team.lower()].copy()
        if panel.empty:
            raise ValueError(f"No rows found for team '{selected_team}' in {args.panel_path}")
    validated_base = build_validated_frame(panel)
    best_config = tune_weights(validated_base, args.train_max_year)
    validated = apply_risk_formula(validated_base, best_config)
    summary = summarize_validation(validated, args.train_max_year, best_config)
    team_summary = summarize_validation_by_team(validated, args.train_max_year)
    deciles = make_deciles(validated)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(summary.to_csv(index=False), encoding="utf-8")
    args.decile_output.write_text(deciles.to_csv(index=False), encoding="utf-8")
    args.scored_output.write_text(validated.to_csv(index=False), encoding="utf-8")
    args.team_summary_output.write_text(team_summary.to_csv(index=False), encoding="utf-8")

    print("Validation complete")
    print(f"- panel rows: {len(validated)}")
    print(f"- summary: {args.summary_output}")
    print(f"- team summary: {args.team_summary_output}")
    print(f"- deciles: {args.decile_output}")
    print(f"- scored: {args.scored_output}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
