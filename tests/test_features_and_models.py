import os
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    calculate_elo_ratings,
    calculate_rolling_features,
    _compute_prior_season_percentiles,
)
from src.models import (
    DixonColesModel,
    blend_predictions,
    apply_temperature_scaling,
)
from src.v2.calibration import (
    build_prematch_calibration_df,
    compute_calibration_curve,
    compute_snapshot_title_trajectories,
)


def _make_synthetic_matches():
    return pd.DataFrame([
        {
            "Season": "2024-2025", "Date": "2024-08-17", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
            "FTHG": 2, "FTAG": 0, "FTR": "H", "HS": 14, "AS": 8, "HST": 6, "AST": 2, "HxG": 2.1, "AxG": 0.6,
        },
        {
            "Season": "2024-2025", "Date": "2024-08-24", "HomeTeam": "Arsenal", "AwayTeam": "Brighton",
            "FTHG": 1, "FTAG": 1, "FTR": "D", "HS": 10, "AS": 10, "HST": 4, "AST": 4, "HxG": 1.3, "AxG": 1.1,
        },
        {
            "Season": "2024-2025", "Date": "2024-08-31", "HomeTeam": "Arsenal", "AwayTeam": "Everton",
            "FTHG": 9, "FTAG": 0, "FTR": "H", "HS": 25, "AS": 2, "HST": 12, "AST": 0, "HxG": 6.5, "AxG": 0.2,
        },
        {
            "Season": "2024-2025", "Date": "2024-09-14", "HomeTeam": "Arsenal", "AwayTeam": "Leeds",
            "FTHG": 1, "FTAG": 0, "FTR": "H", "HS": 12, "AS": 6, "HST": 5, "AST": 2, "HxG": 1.4, "AxG": 0.5,
        },
    ])


def test_rolling_features_zero_future_leakage():
    df = _make_synthetic_matches()
    out = calculate_rolling_features(df, rolling_window=10)

    # Match 1 (no history) uses neutral defaults
    assert out.loc[0, "HomeRollingGoalsScored"] == pytest.approx(1.3)
    assert out.loc[0, "HomeRollingXG"] == pytest.approx(1.3)

    # Match 3 (the 9-0 blowup, HxG=6.5) must ONLY see Matches 1-2: avg goals=(2+1)/2=1.5, avg xG=(2.1+1.3)/2=1.7, avg pts=(3+1)/2=2.0
    assert out.loc[2, "HomeRollingGoalsScored"] == pytest.approx(1.5)
    assert out.loc[2, "HomeRollingXG"] == pytest.approx(1.7)
    assert out.loc[2, "HomeRollingPoints"] == pytest.approx(2.0)

    # Match 4 sees Matches 1-3: avg goals=(2+1+9)/3=4.0, avg xG=(2.1+1.3+6.5)/3=3.3
    assert out.loc[3, "HomeRollingGoalsScored"] == pytest.approx(4.0)
    assert out.loc[3, "HomeRollingXG"] == pytest.approx(3.3)


def test_elo_ratings_prematch_order_and_conservation():
    df = _make_synthetic_matches()
    out, final_elos, _ = calculate_elo_ratings(df, k_factor=20, hfa=100)

    # First match starts at 1500 vs 1500
    assert out.loc[0, "HomeElo"] == pytest.approx(1500.0)
    assert out.loc[0, "AwayElo"] == pytest.approx(1500.0)

    # Within a single season, sum of all team Elos stays invariant (zero-sum updates)
    total_elo = sum(final_elos.values())
    assert total_elo == pytest.approx(len(final_elos) * 1500.0, abs=1e-6)


def test_prior_season_percentiles_zero_within_season_leakage():
    df = pd.DataFrame([
        {"Season": "2023-2024", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTR": "H"},
        {"Season": "2023-2024", "HomeTeam": "Chelsea", "AwayTeam": "Everton", "FTR": "H"},
        {"Season": "2024-2025", "HomeTeam": "Arsenal", "AwayTeam": "Everton", "FTR": "A"},
    ])
    pct_before = _compute_prior_season_percentiles(df)

    # Flip 2024-2025 match outcome; 2024-2025 prior-season percentiles must NOT change
    df_flipped = df.copy()
    df_flipped.loc[2, "FTR"] = "H"
    pct_after = _compute_prior_season_percentiles(df_flipped)

    assert pct_before[("2024-2025", "Arsenal")] == pytest.approx(pct_after[("2024-2025", "Arsenal")])
    assert pct_before[("2024-2025", "Everton")] == pytest.approx(pct_after[("2024-2025", "Everton")])
    assert pct_before[("2024-2025", "Arsenal")] > pct_before[("2024-2025", "Everton")]


def test_dixon_coles_probability_sum_and_orientation():
    # Construct model with strong home team vs weak away team so HomeWin must far exceed AwayWin
    dc = DixonColesModel()
    dc.team_indices = {"Strong": 0, "Weak": 1}
    dc.attack_params = np.array([2.0, 0.5])
    dc.defense_params = np.array([0.5, 2.0])
    dc.home_advantage = 1.25
    dc.rho = -0.04

    res = dc.predict_match_probabilities("Strong", "Weak")
    total = res["HomeWin"] + res["Draw"] + res["AwayWin"]
    assert total == pytest.approx(1.0, abs=1e-6)
    assert res["HomeWin"] > 0.80
    assert res["HomeWin"] > res["AwayWin"]

    # Reverse fixture: Weak at home vs Strong away -> AwayWin must exceed HomeWin
    res_rev = dc.predict_match_probabilities("Weak", "Strong")
    assert res_rev["HomeWin"] + res_rev["Draw"] + res_rev["AwayWin"] == pytest.approx(1.0, abs=1e-6)
    assert res_rev["AwayWin"] > res_rev["HomeWin"]


def test_blend_and_temperature_scaling_normalization():
    p_dc = {"HomeWin": 0.55, "Draw": 0.25, "AwayWin": 0.20}
    p_xgb = {"HomeWin": 0.45, "Draw": 0.30, "AwayWin": 0.25}
    blended = blend_predictions(p_dc, p_xgb, dc_weight=0.35)
    assert blended["HomeWin"] + blended["Draw"] + blended["AwayWin"] == pytest.approx(1.0, abs=1e-8)

    arr_1d = np.array([0.20, 0.30, 0.50])
    cal_1d = apply_temperature_scaling(arr_1d, temperature=1.15)
    assert cal_1d.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.argmax(cal_1d) == 2

    arr_2d = np.array([[0.20, 0.30, 0.50], [0.60, 0.25, 0.15]])
    cal_2d = apply_temperature_scaling(arr_2d, temperature=1.2)
    assert np.allclose(cal_2d.sum(axis=1), 1.0, atol=1e-8)


def test_output_predictions_probabilities_sum_to_100():
    preds_path = "output/predictions.csv"
    if not os.path.exists(preds_path):
        pytest.skip("output/predictions.csv not generated")
    preds = pd.read_csv(preds_path)
    totals = preds["HomeWin%"] + preds["Draw%"] + preds["AwayWin%"]
    assert np.allclose(totals, 100.0, atol=0.1)


def test_calibration_utilities_smoke():
    hist = pd.DataFrame([
        {
            "MatchDay": 1, "Date": "2026-08-21", "HomeTeam": "Arsenal", "AwayTeam": "Coventry",
            "HomeWin%": 70.0, "Draw%": 20.0, "AwayWin%": 10.0, "MostLikelyScore": "2-0",
            "SnapshotDate": "2026-08-20 10:00:00", "CompletedMatchesCount": 0, "Status": "Scheduled", "ActualScore": None,
        },
        {
            "MatchDay": 1, "Date": "2026-08-22", "HomeTeam": "Hull", "AwayTeam": "Man United",
            "HomeWin%": 25.0, "Draw%": 25.0, "AwayWin%": 50.0, "MostLikelyScore": "1-2",
            "SnapshotDate": "2026-08-20 10:00:00", "CompletedMatchesCount": 0, "Status": "Scheduled", "ActualScore": None,
        },
    ])
    completed = pd.DataFrame([
        {"Date": "2026-08-21", "HomeTeam": "Arsenal", "AwayTeam": "Coventry", "FTHG": 3, "FTAG": 0, "FTR": "H"},
        {"Date": "2026-08-22", "HomeTeam": "Hull", "AwayTeam": "Man United", "FTHG": 2, "FTAG": 0, "FTR": "H"},
    ])
    calib = build_prematch_calibration_df(hist, completed)
    assert len(calib) == 2
    assert calib.loc[0, "Correct"] == 1
    assert calib.loc[1, "Correct"] == 0
    curve = compute_calibration_curve(calib)
    assert not curve.empty
    traj = compute_snapshot_title_trajectories(hist, completed, num_sims=50)
    assert not traj.empty

