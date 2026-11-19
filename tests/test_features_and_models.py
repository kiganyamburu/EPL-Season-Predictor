import os
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    calculate_elo_ratings,
    calculate_rolling_features,
    identify_promoted_teams,
    add_market_values_and_managers,
    _compute_prior_season_percentiles,
)
from src.simulation import run_season_simulations
from src.models import (
    DixonColesModel,
    XGBoostPredictor,
    blend_predictions,
    apply_temperature_scaling,
    fit_temperature_scaler,
)
from src.pipeline_v2 import _infer_target_season, fit_dynamic_temperature
from src.v2.config import get_v2_config, load_calibrated_temperature, CALIBRATION_TEMP
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


def test_fit_temperature_scaler_detects_over_and_under_confidence():
    # Overconfident predictions (extreme confidence despite errors) -> T > 1.0 lowers NLL
    overconf = np.array([
        [0.01, 0.04, 0.95],
        [0.95, 0.04, 0.01],
        [0.02, 0.03, 0.95],
        [0.95, 0.03, 0.02],
        [0.01, 0.04, 0.95],
        [0.95, 0.04, 0.01],
    ])
    y_over = np.array([2, 0, 1, 2, 0, 1])
    t_over = fit_temperature_scaler(overconf, y_over)
    assert t_over > 1.2
    cal_over = apply_temperature_scaling(overconf, t_over)
    nll_before = -np.mean(np.log(overconf[np.arange(len(y_over)), y_over]))
    nll_after = -np.mean(np.log(cal_over[np.arange(len(y_over)), y_over]))
    assert nll_after < nll_before

    # Underconfident predictions (true outcomes consistently match argmax, probabilities overly flat) -> T < 1.0
    underconf = np.array([
        [0.30, 0.30, 0.40],
        [0.42, 0.30, 0.28],
        [0.25, 0.45, 0.30],
        [0.28, 0.31, 0.41],
        [0.44, 0.28, 0.28],
        [0.27, 0.46, 0.27],
    ])
    y_under = np.array([2, 0, 1, 2, 0, 1])
    t_under = fit_temperature_scaler(underconf, y_under)
    assert t_under < 0.85
    cal_under = apply_temperature_scaling(underconf, t_under)
    nll_u_before = -np.mean(np.log(underconf[np.arange(len(y_under)), y_under]))
    nll_u_after = -np.mean(np.log(cal_under[np.arange(len(y_under)), y_under]))
    assert nll_u_after < nll_u_before

    # Uniform predictions should remain at T = 1.0
    uniform = np.full((6, 3), 1.0 / 3.0)
    assert fit_temperature_scaler(uniform, y_over) == pytest.approx(1.0)


def test_dynamic_calibration_config_and_target_season_inference(tmp_path):
    fixtures = pd.DataFrame([
        {"Date": "2026-08-21", "HomeTeam": "Arsenal", "AwayTeam": "Leeds"},
        {"Date": "2027-05-23", "HomeTeam": "Chelsea", "AwayTeam": "Arsenal"},
    ])
    assert _infer_target_season(fixtures) == "2026-2027"

    # When no prior validation season meets min_val_matches, fallback temperature is preserved
    small_hist = _make_synthetic_matches()
    cfg = get_v2_config(use_saved_calibration=False)
    cal = fit_dynamic_temperature(small_hist, cfg, target_season="2025-2026", min_val_matches=50)
    assert cal["validation_season"] is None
    assert cal["calibration_temp"] == pytest.approx(CALIBRATION_TEMP)

    # Reading persisted calibration params JSON works cleanly and ignores invalid files
    cal_file = tmp_path / "cal.json"
    cal_file.write_text('{"calibration_temp": 1.24}', encoding="utf-8")
    assert load_calibrated_temperature(path=str(cal_file)) == pytest.approx(1.24)
    cal_file.write_text('{"calibration_temp": 99.0}', encoding="utf-8")
    assert load_calibrated_temperature(path=str(cal_file)) == pytest.approx(CALIBRATION_TEMP)


def test_rolling_sot_and_red_cards_zero_leakage():
    df = _make_synthetic_matches().copy()
    df["HR"] = [0, 1, 0, 0]
    df["AR"] = [0, 0, 1, 0]
    out = calculate_rolling_features(df, rolling_window=10)

    # Match 1 sees neutral defaults
    assert out.loc[0, "HomeRollingSOTFor"] == pytest.approx(4.4)
    assert out.loc[0, "HomeRollingRedCards"] == pytest.approx(0.06)

    # Match 3 (HST=12) sees only Matches 1-2: SOTFor=(6+4)/2=5.0, SOTAgainst=(2+4)/2=3.0, RedCards=(0+1)/2=0.5
    assert out.loc[2, "HomeRollingSOTFor"] == pytest.approx(5.0)
    assert out.loc[2, "HomeRollingSOTAgainst"] == pytest.approx(3.0)
    assert out.loc[2, "HomeRollingRedCards"] == pytest.approx(0.5)

    # Match 4 sees Matches 1-3: SOTFor=(6+4+12)/3=22/3, RedCards=(0+1+0)/3=1/3
    assert out.loc[3, "HomeRollingSOTFor"] == pytest.approx(22.0 / 3.0)
    assert out.loc[3, "HomeRollingRedCards"] == pytest.approx(1.0 / 3.0)


def test_historical_manager_multi_spell_and_transfer_proxies(tmp_path):
    mgr_csv = tmp_path / "hist_mgr.csv"
    mgr_csv.write_text(
        "Team,Manager,AppointedDate\n"
        "Chelsea,Antonio Conte,2016-07-01\n"
        "Chelsea,Thomas Tuchel,2021-01-26\n",
        encoding="utf-8",
    )
    matches = pd.DataFrame([
        {"Season": "2016-2017", "Date": "2016-08-20", "HomeTeam": "Chelsea", "AwayTeam": "Arsenal", "FTR": "H"},
        {"Season": "2017-2018", "Date": "2018-03-01", "HomeTeam": "Chelsea", "AwayTeam": "Arsenal", "FTR": "D"},
        {"Season": "2020-2021", "Date": "2021-02-15", "HomeTeam": "Chelsea", "AwayTeam": "Arsenal", "FTR": "H"},
    ])
    enriched = add_market_values_and_managers(
        matches,
        squad_values_path=str(tmp_path / "missing_squads.csv"),
        managers_path=str(mgr_csv),
        historical_managers_path=None,
    )
    # 50 days into Conte spell -> 1; >600 days -> 0; 20 days into Tuchel spell -> 1
    assert list(enriched["HomeNewManager"]) == [1, 0, 1]
    assert set(XGBoostPredictor().features).issuperset({
        "HomeTransferImpact", "AwayTransferImpact", "HomeNetSpend", "AwayNetSpend"
    })
    # Season 2017-2018 Chelsea finished ahead of Arsenal in synthetic 2016-2017 -> higher NetSpend proxy
    assert enriched.loc[1, "HomeNetSpend"] > enriched.loc[1, "AwayNetSpend"]


def test_dixon_coles_xg_blend_uses_xg_signal():
    # Three clubs with identical 1-1 integer goals, where TeamA creates high xG vs peers
    rows = []
    for dt in ["2024-08-17", "2024-08-24", "2024-08-31", "2024-09-14", "2024-09-21"]:
        rows.extend([
            {"Date": dt, "HomeTeam": "TeamA", "AwayTeam": "TeamB", "FTHG": 1, "FTAG": 1, "HxG": 3.2, "AxG": 0.3},
            {"Date": dt, "HomeTeam": "TeamC", "AwayTeam": "TeamA", "FTHG": 1, "FTAG": 1, "HxG": 0.3, "AxG": 3.2},
            {"Date": dt, "HomeTeam": "TeamB", "AwayTeam": "TeamC", "FTHG": 1, "FTAG": 1, "HxG": 1.1, "AxG": 1.1},
        ])
    df = pd.DataFrame(rows)

    dc_goals = DixonColesModel(time_decay_phi=0.0, xg_weight=0.0)
    dc_goals.fit(df)
    a_idx, b_idx = dc_goals.team_indices["TeamA"], dc_goals.team_indices["TeamB"]
    assert dc_goals.attack_params[a_idx] == pytest.approx(dc_goals.attack_params[b_idx], abs=1e-2)

    dc_xg = DixonColesModel(time_decay_phi=0.0, xg_weight=0.4)
    dc_xg.fit(df)
    ax, bx = dc_xg.team_indices["TeamA"], dc_xg.team_indices["TeamB"]
    assert dc_xg.attack_params[ax] > dc_xg.attack_params[bx] * 1.5


def test_v1_simulation_post_gw10_state_and_matchday_integrity():
    # Minimal 2-season dataset with H/D/A outcomes and completed + scheduled 2026/27 fixtures
    rows = [
        {"Season": "2025-2026", "Date": "2025-08-16", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTHG": 2, "FTAG": 0, "FTR": "H", "HS": 15, "AS": 7, "HST": 6, "AST": 2, "HR": 0, "AR": 0},
        {"Season": "2025-2026", "Date": "2025-08-23", "HomeTeam": "Chelsea", "AwayTeam": "Liverpool", "FTHG": 1, "FTAG": 1, "FTR": "D", "HS": 10, "AS": 10, "HST": 4, "AST": 4, "HR": 0, "AR": 0},
        {"Season": "2025-2026", "Date": "2025-08-30", "HomeTeam": "Liverpool", "AwayTeam": "Arsenal", "FTHG": 0, "FTAG": 2, "FTR": "A", "HS": 8, "AS": 14, "HST": 3, "AST": 6, "HR": 0, "AR": 0},
        {"Season": "2026-2027", "Date": "2026-08-22", "HomeTeam": "Arsenal", "AwayTeam": "Coventry", "FTHG": 3, "FTAG": 0, "FTR": "H", "HS": 18, "AS": 4, "HST": 8, "AST": 1, "HR": 0, "AR": 0, "HxG": 2.8, "AxG": 0.4},
        {"Season": "2026-2027", "Date": "2026-08-29", "HomeTeam": "Chelsea", "AwayTeam": "Liverpool", "FTHG": 1, "FTAG": 2, "FTR": "A", "HS": 9, "AS": 13, "HST": 3, "AST": 5, "HR": 0, "AR": 0, "HxG": 0.9, "AxG": 2.1},
    ]
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df, final_elos, _ = calculate_elo_ratings(df)
    df = calculate_rolling_features(df)
    df = identify_promoted_teams(df)
    df = add_market_values_and_managers(df)

    fixtures = pd.DataFrame([
        {"MatchDay": 1, "Date": "2026-08-22", "HomeTeam": "Arsenal", "AwayTeam": "Coventry", "Summary": "Arsenal - Coventry (3-0)"},
        {"MatchDay": 2, "Date": "2026-08-29", "HomeTeam": "Chelsea", "AwayTeam": "Liverpool", "Summary": "Chelsea - Liverpool (1-2)"},
        {"MatchDay": 11, "Date": "2026-11-21", "HomeTeam": "Coventry", "AwayTeam": "Chelsea", "Summary": "Coventry - Chelsea"},
    ])
    summary = run_season_simulations(fixtures, final_elos, df, num_sims=20, save_outputs=False)
    assert set(summary["Team"]) == {"Arsenal", "Chelsea", "Liverpool", "Coventry"}
    # Arsenal & Liverpool already have 3 guaranteed points from completed GW1/GW2; Chelsea played 2 matches
    pts_map = dict(zip(summary["Team"], summary["ExpectedPoints"]))
    assert pts_map["Arsenal"] == pytest.approx(3.0)
    assert pts_map["Liverpool"] == pytest.approx(3.0)



