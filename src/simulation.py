import bisect
import os
import re
import numpy as np
import pandas as pd
from scipy.special import gammaln

from src.feature_engineering import _estimate_match_xg, _load_manager_spell_lookup
from src.models import DixonColesModel, XGBoostPredictor, blend_predictions
from src.pipeline_v2 import _build_xgb_features, _load_transfer_lookup


def run_season_simulations(
    fixtures_df,
    final_elos,
    df_historical,
    managers_path='data/raw/managers.csv',
    num_sims=10000,
    dc_weight=0.5,
    save_outputs=True,
):
    """
    Run stateful Monte Carlo simulations of the 2026/27 EPL season (v1 baseline ensemble).
    Initializes from post-GW10 live state when 2026-2027 matches exist, tracks dynamic
    rolling buffers (goals, xG/xGA, shots on target, red cards, points) across trajectories,
    and respects explicit MatchDay scheduling.
    """
    print(f"=== INITIALIZING MONTE CARLO SIMULATOR ({num_sims} RUNS) ===")

    df_historical = df_historical.copy()
    df_historical['Date'] = pd.to_datetime(df_historical['Date'], format='mixed', dayfirst=True)

    # 1. Fit Dixon-Coles and XGBoost on full historical + completed 2026/27 data
    dc_model = DixonColesModel()
    dc_model.fit(df_historical)

    xgb_predictor = XGBoostPredictor()
    xgb_predictor.features = _build_xgb_features(
        df_historical,
        {'promoted_handling': True, 'congestion': False, 'injuries': False},
    )
    xgb_predictor.fit(df_historical)
    feature_names = xgb_predictor.features

    # 2. Load club metadata (squad values, historical + current manager spells, transfers)
    squad_df = pd.read_csv('data/raw/squad_values.csv')
    squad_values = dict(zip(squad_df['Team'], squad_df['MarketValue_M_Euros']))
    max_val = max(squad_values.values())
    min_val = min(squad_values.values())

    manager_spells = _load_manager_spell_lookup(managers_path, 'data/raw/historical_managers.csv')
    transfer_impacts, net_spends = _load_transfer_lookup()

    fixtures_df = fixtures_df.copy()
    fixtures_df['Date'] = pd.to_datetime(fixtures_df['Date'])
    fixtures_df = fixtures_df.sort_values('Date').reset_index(drop=True)

    active_teams = sorted(set(fixtures_df['HomeTeam']).union(set(fixtures_df['AwayTeam'])))
    num_teams = len(active_teams)
    team_to_idx = {team: i for i, team in enumerate(active_teams)}

    teams_2526 = set(df_historical[df_historical['Season'] == '2025-2026']['HomeTeam']).union(
        set(df_historical[df_historical['Season'] == '2025-2026']['AwayTeam'])
    )
    relegated_2526 = teams_2526 - set(active_teams)
    promoted_2627 = set(active_teams) - teams_2526

    if relegated_2526:
        promoted_baseline = float(np.mean([final_elos.get(t, 1500.0) for t in relegated_2526]))
    else:
        promoted_baseline = 1420.0

    # Initialize Elo from live post-completed-match state when season is underway
    df_2627 = df_historical[df_historical['Season'] == '2026-2027'].sort_values('Date')
    in_season = not df_2627.empty
    hist_2627_records = {}
    if in_season:
        baseline_elos = {t: float(final_elos.get(t, 1500.0)) for t in active_teams}
        for _, hrow in df_2627.iterrows():
            hist_2627_records[(hrow['HomeTeam'], hrow['AwayTeam'])] = hrow
    else:
        baseline_elos = {
            t: (0.7 * float(final_elos[t]) + 0.3 * 1500.0 if (t in final_elos and t not in promoted_2627) else promoted_baseline)
            for t in active_teams
        }

    elo_sims = np.zeros((num_sims, num_teams), dtype=float)
    for t_idx, team in enumerate(active_teams):
        elo_sims[:, t_idx] = baseline_elos[team]

    # 3. Seed 10-match circular rolling buffers from df_historical (including HST/AST, HR/AR, HxG/AxG)
    ROLL_WIN = 10
    roll_gf = np.zeros((num_sims, num_teams, ROLL_WIN), dtype=np.float32)
    roll_ga = np.zeros((num_sims, num_teams, ROLL_WIN), dtype=np.float32)
    roll_xg = np.zeros((num_sims, num_teams, ROLL_WIN), dtype=np.float32)
    roll_xga = np.zeros((num_sims, num_teams, ROLL_WIN), dtype=np.float32)
    roll_sot_for = np.zeros((num_sims, num_teams, ROLL_WIN), dtype=np.float32)
    roll_sot_against = np.zeros((num_sims, num_teams, ROLL_WIN), dtype=np.float32)
    roll_red = np.zeros((num_sims, num_teams, ROLL_WIN), dtype=np.float32)
    roll_pts = np.zeros((num_sims, num_teams, ROLL_WIN), dtype=np.float32)
    roll_len = np.zeros(num_teams, dtype=int)
    roll_pos = np.zeros(num_teams, dtype=int)

    df_sorted = df_historical.sort_values('Date')
    for t_idx, team in enumerate(active_teams):
        tm = df_sorted[(df_sorted['HomeTeam'] == team) | (df_sorted['AwayTeam'] == team)].tail(ROLL_WIN)
        k = len(tm)
        if k > 0:
            gfs, gas, xgs, xgas, sfs, sas, reds, ptss = [], [], [], [], [], [], [], []
            for _, mr in tm.iterrows():
                h_xg_m = _estimate_match_xg(mr, 'H')
                a_xg_m = _estimate_match_xg(mr, 'A')
                h_sot_m = float(mr['HST']) if ('HST' in mr and pd.notna(mr['HST'])) else 4.4
                a_sot_m = float(mr['AST']) if ('AST' in mr and pd.notna(mr['AST'])) else 4.4
                h_red_m = float(mr['HR']) if ('HR' in mr and pd.notna(mr['HR'])) else 0.0
                a_red_m = float(mr['AR']) if ('AR' in mr and pd.notna(mr['AR'])) else 0.0
                if mr['HomeTeam'] == team:
                    hg_m, ag_m = float(mr['FTHG']), float(mr['FTAG'])
                    xg_m, xga_m = h_xg_m, a_xg_m
                    sf_m, sa_m, rd_m = h_sot_m, a_sot_m, h_red_m
                    p_m = 3.0 if mr['FTR'] == 'H' else (1.0 if mr['FTR'] == 'D' else 0.0)
                else:
                    hg_m, ag_m = float(mr['FTAG']), float(mr['FTHG'])
                    xg_m, xga_m = a_xg_m, h_xg_m
                    sf_m, sa_m, rd_m = a_sot_m, h_sot_m, a_red_m
                    p_m = 3.0 if mr['FTR'] == 'A' else (1.0 if mr['FTR'] == 'D' else 0.0)
                gfs.append(hg_m); gas.append(ag_m)
                xgs.append(xg_m); xgas.append(xga_m)
                sfs.append(sf_m); sas.append(sa_m)
                reds.append(rd_m); ptss.append(p_m)
            roll_gf[:, t_idx, :k] = np.array(gfs, dtype=np.float32)
            roll_ga[:, t_idx, :k] = np.array(gas, dtype=np.float32)
            roll_xg[:, t_idx, :k] = np.array(xgs, dtype=np.float32)
            roll_xga[:, t_idx, :k] = np.array(xgas, dtype=np.float32)
            roll_sot_for[:, t_idx, :k] = np.array(sfs, dtype=np.float32)
            roll_sot_against[:, t_idx, :k] = np.array(sas, dtype=np.float32)
            roll_red[:, t_idx, :k] = np.array(reds, dtype=np.float32)
            roll_pts[:, t_idx, :k] = np.array(ptss, dtype=np.float32)
            roll_len[t_idx] = k
            roll_pos[t_idx] = k % ROLL_WIN

    # 4. Chronological vectorized season simulation loop
    max_goals = 10
    num_scores = (max_goals + 1) ** 2
    log_fact = gammaln(np.arange(max_goals + 1) + 1)
    score_to_goals = np.array([(hg, ag) for hg in range(max_goals + 1) for ag in range(max_goals + 1)])
    x_coords, y_coords = np.meshgrid(np.arange(max_goals + 1), np.arange(max_goals + 1), indexing='ij')
    mask_h = x_coords > y_coords
    mask_d = x_coords == y_coords
    mask_a = x_coords < y_coords

    points_sims = np.zeros((num_sims, num_teams), dtype=int)
    gd_sims = np.zeros((num_sims, num_teams), dtype=int)
    gf_sims = np.zeros((num_sims, num_teams), dtype=int)
    ranks_sims = np.zeros((num_sims, num_teams), dtype=int)
    preds_output = []

    def _is_new_mgr(team_name, match_dt):
        spells = manager_spells.get(team_name)
        if spells:
            pos = bisect.bisect_right(spells, match_dt) - 1
            if pos >= 0:
                return 1 if 0 <= (match_dt - spells[pos]).days < 180 else 0
        return 0

    print("Running chronological Monte Carlo simulation...")
    for idx, row in fixtures_df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        date = row['Date']
        matchday = int(row['MatchDay']) if 'MatchDay' in row and pd.notna(row['MatchDay']) else (idx // 10 + 1)
        h_idx = team_to_idx[home]
        a_idx = team_to_idx[away]

        h_val_pct = (squad_values.get(home, 100.0) - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        a_val_pct = (squad_values.get(away, 100.0) - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        h_new_mgr = _is_new_mgr(home, date)
        a_new_mgr = _is_new_mgr(away, date)
        h_prom = 1 if home in promoted_2627 else 0
        a_prom = 1 if away in promoted_2627 else 0
        h_t_imp = transfer_impacts.get(home, 1.14)
        a_t_imp = transfer_impacts.get(away, 1.14)
        h_t_ns = net_spends.get(home, 174.0)
        a_t_ns = net_spends.get(away, 174.0)

        h_idx_dc = dc_model.team_indices.get(home)
        a_idx_dc = dc_model.team_indices.get(away)
        if h_idx_dc is not None and a_idx_dc is not None:
            lmbda_base = dc_model.attack_params[h_idx_dc] * dc_model.defense_params[a_idx_dc] * dc_model.home_advantage
            mu_base = dc_model.attack_params[a_idx_dc] * dc_model.defense_params[h_idx_dc]
        else:
            lmbda_base, mu_base = 1.35, 1.15

        nh = int(roll_len[h_idx])
        na = int(roll_len[a_idx])
        h_r_gf = roll_gf[:, h_idx, :nh].mean(axis=1) if nh > 0 else np.full(num_sims, 1.3, dtype=np.float32)
        h_r_ga = roll_ga[:, h_idx, :nh].mean(axis=1) if nh > 0 else np.full(num_sims, 1.3, dtype=np.float32)
        h_r_xg = roll_xg[:, h_idx, :nh].mean(axis=1) if nh > 0 else np.full(num_sims, 1.3, dtype=np.float32)
        h_r_xga = roll_xga[:, h_idx, :nh].mean(axis=1) if nh > 0 else np.full(num_sims, 1.3, dtype=np.float32)
        h_r_sf = roll_sot_for[:, h_idx, :nh].mean(axis=1) if nh > 0 else np.full(num_sims, 4.4, dtype=np.float32)
        h_r_sa = roll_sot_against[:, h_idx, :nh].mean(axis=1) if nh > 0 else np.full(num_sims, 4.4, dtype=np.float32)
        h_r_rd = roll_red[:, h_idx, :nh].mean(axis=1) if nh > 0 else np.full(num_sims, 0.06, dtype=np.float32)
        h_r_pts = roll_pts[:, h_idx, :nh].mean(axis=1) if nh > 0 else np.full(num_sims, 1.1, dtype=np.float32)

        a_r_gf = roll_gf[:, a_idx, :na].mean(axis=1) if na > 0 else np.full(num_sims, 1.3, dtype=np.float32)
        a_r_ga = roll_ga[:, a_idx, :na].mean(axis=1) if na > 0 else np.full(num_sims, 1.3, dtype=np.float32)
        a_r_xg = roll_xg[:, a_idx, :na].mean(axis=1) if na > 0 else np.full(num_sims, 1.3, dtype=np.float32)
        a_r_xga = roll_xga[:, a_idx, :na].mean(axis=1) if na > 0 else np.full(num_sims, 1.3, dtype=np.float32)
        a_r_sf = roll_sot_for[:, a_idx, :na].mean(axis=1) if na > 0 else np.full(num_sims, 4.4, dtype=np.float32)
        a_r_sa = roll_sot_against[:, a_idx, :na].mean(axis=1) if na > 0 else np.full(num_sims, 4.4, dtype=np.float32)
        a_r_rd = roll_red[:, a_idx, :na].mean(axis=1) if na > 0 else np.full(num_sims, 0.06, dtype=np.float32)
        a_r_pts = roll_pts[:, a_idx, :na].mean(axis=1) if na > 0 else np.full(num_sims, 1.1, dtype=np.float32)

        summary_str = str(row.get('Summary', ''))
        m_score = re.search(r'\((\d+)-(\d+)\)', summary_str)
        if m_score:
            actual_hg = int(m_score.group(1))
            actual_ag = int(m_score.group(2))
            h_pts = 3 if actual_hg > actual_ag else (1 if actual_hg == actual_ag else 0)
            a_pts = 3 if actual_ag > actual_hg else (1 if actual_hg == actual_ag else 0)
            points_sims[:, h_idx] += h_pts
            points_sims[:, a_idx] += a_pts
            gf_sims[:, h_idx] += actual_hg
            gf_sims[:, a_idx] += actual_ag
            gd_sims[:, h_idx] += actual_hg - actual_ag
            gd_sims[:, a_idx] += actual_ag - actual_hg

            hist_row = hist_2627_records.get((home, away))
            if hist_row is None:
                # Completed in fixtures_df before feature_engineering sync: advance Elo once
                e_h_s = 1.0 / (1.0 + 10.0 ** ((elo_sims[:, a_idx] - elo_sims[:, h_idx] - 100.0) / 400.0))
                s_h_s = 1.0 if actual_hg > actual_ag else (0.5 if actual_hg == actual_ag else 0.0)
                elo_sims[:, h_idx] += 20.0 * (s_h_s - e_h_s)
                elo_sims[:, a_idx] += 20.0 * ((1.0 - s_h_s) - (1.0 - e_h_s))

            if save_outputs:
                if hist_row is not None:
                    feat_1 = {
                        'EloDiff': float(hist_row['HomeElo']) - float(hist_row['AwayElo']),
                        'HomeRollingGoalsScored': float(hist_row['HomeRollingGoalsScored']),
                        'HomeRollingGoalsConceded': float(hist_row['HomeRollingGoalsConceded']),
                        'HomeRollingPoints': float(hist_row['HomeRollingPoints']),
                        'HomeRollingXG': float(hist_row.get('HomeRollingXG', hist_row['HomeRollingGoalsScored'])),
                        'HomeRollingXGA': float(hist_row.get('HomeRollingXGA', hist_row['HomeRollingGoalsConceded'])),
                        'HomeRollingSOTFor': float(hist_row.get('HomeRollingSOTFor', 4.4)),
                        'HomeRollingSOTAgainst': float(hist_row.get('HomeRollingSOTAgainst', 4.4)),
                        'HomeRollingRedCards': float(hist_row.get('HomeRollingRedCards', 0.06)),
                        'AwayRollingGoalsScored': float(hist_row['AwayRollingGoalsScored']),
                        'AwayRollingGoalsConceded': float(hist_row['AwayRollingGoalsConceded']),
                        'AwayRollingPoints': float(hist_row['AwayRollingPoints']),
                        'AwayRollingXG': float(hist_row.get('AwayRollingXG', hist_row['AwayRollingGoalsScored'])),
                        'AwayRollingXGA': float(hist_row.get('AwayRollingXGA', hist_row['AwayRollingGoalsConceded'])),
                        'AwayRollingSOTFor': float(hist_row.get('AwayRollingSOTFor', 4.4)),
                        'AwayRollingSOTAgainst': float(hist_row.get('AwayRollingSOTAgainst', 4.4)),
                        'AwayRollingRedCards': float(hist_row.get('AwayRollingRedCards', 0.06)),
                        'HomeTransferImpact': float(hist_row.get('HomeTransferImpact', h_t_imp)),
                        'AwayTransferImpact': float(hist_row.get('AwayTransferImpact', a_t_imp)),
                        'HomeNetSpend': float(hist_row.get('HomeNetSpend', h_t_ns)),
                        'AwayNetSpend': float(hist_row.get('AwayNetSpend', a_t_ns)),
                    }
                else:
                    feat_1 = {
                        'EloDiff': float(elo_sims[:, h_idx].mean() - elo_sims[:, a_idx].mean()),
                        'HomeRollingGoalsScored': float(h_r_gf.mean()),
                        'HomeRollingGoalsConceded': float(h_r_ga.mean()),
                        'HomeRollingPoints': float(h_r_pts.mean()),
                        'HomeRollingXG': float(h_r_xg.mean()),
                        'HomeRollingXGA': float(h_r_xga.mean()),
                        'HomeRollingSOTFor': float(h_r_sf.mean()),
                        'HomeRollingSOTAgainst': float(h_r_sa.mean()),
                        'HomeRollingRedCards': float(h_r_rd.mean()),
                        'AwayRollingGoalsScored': float(a_r_gf.mean()),
                        'AwayRollingGoalsConceded': float(a_r_ga.mean()),
                        'AwayRollingPoints': float(a_r_pts.mean()),
                        'AwayRollingXG': float(a_r_xg.mean()),
                        'AwayRollingXGA': float(a_r_xga.mean()),
                        'AwayRollingSOTFor': float(a_r_sf.mean()),
                        'AwayRollingSOTAgainst': float(a_r_sa.mean()),
                        'AwayRollingRedCards': float(a_r_rd.mean()),
                        'HomeTransferImpact': h_t_imp,
                        'AwayTransferImpact': a_t_imp,
                        'HomeNetSpend': h_t_ns,
                        'AwayNetSpend': a_t_ns,
                    }
                feat_1.update({
                    'HomeSquadValPercentile': h_val_pct,
                    'AwaySquadValPercentile': a_val_pct,
                    'HomeNewManager': h_new_mgr,
                    'AwayNewManager': a_new_mgr,
                    'HomePromoted': h_prom,
                    'AwayPromoted': a_prom,
                })
                x_1 = np.array([[feat_1[c] for c in feature_names]], dtype=np.float32)
                xp_1 = xgb_predictor.model.predict_proba(x_1)[0]
                dc_1 = dc_model.predict_match_probabilities(home, away, max_goals=max_goals)
                blended = blend_predictions(
                    dc_1,
                    {'HomeWin': float(xp_1[2]), 'Draw': float(xp_1[1]), 'AwayWin': float(xp_1[0])},
                    dc_weight=dc_weight,
                )
                preds_output.append({
                    'MatchDay': matchday,
                    'Date': date.strftime('%Y-%m-%d'),
                    'HomeTeam': home,
                    'AwayTeam': away,
                    'Status': 'Completed',
                    'ActualScore': f"{actual_hg}-{actual_ag}",
                    'HomeWin%': round(blended['HomeWin'] * 100, 2),
                    'Draw%': round(blended['Draw'] * 100, 2),
                    'AwayWin%': round(blended['AwayWin'] * 100, 2),
                    'MostLikelyScore': f"{actual_hg}-{actual_ag}",
                })
            continue

        # Unplayed future fixture: dynamic per-trajectory features + Elo updates
        h_elos = elo_sims[:, h_idx]
        a_elos = elo_sims[:, a_idx]
        col_map = {
            'EloDiff': (h_elos - a_elos).astype(np.float32),
            'HomeRollingGoalsScored': h_r_gf,
            'HomeRollingGoalsConceded': h_r_ga,
            'HomeRollingPoints': h_r_pts,
            'HomeRollingXG': h_r_xg,
            'HomeRollingXGA': h_r_xga,
            'HomeRollingSOTFor': h_r_sf,
            'HomeRollingSOTAgainst': h_r_sa,
            'HomeRollingRedCards': h_r_rd,
            'AwayRollingGoalsScored': a_r_gf,
            'AwayRollingGoalsConceded': a_r_ga,
            'AwayRollingPoints': a_r_pts,
            'AwayRollingXG': a_r_xg,
            'AwayRollingXGA': a_r_xga,
            'AwayRollingSOTFor': a_r_sf,
            'AwayRollingSOTAgainst': a_r_sa,
            'AwayRollingRedCards': a_r_rd,
            'HomeSquadValPercentile': np.full(num_sims, h_val_pct, dtype=np.float32),
            'AwaySquadValPercentile': np.full(num_sims, a_val_pct, dtype=np.float32),
            'HomeTransferImpact': np.full(num_sims, h_t_imp, dtype=np.float32),
            'AwayTransferImpact': np.full(num_sims, a_t_imp, dtype=np.float32),
            'HomeNetSpend': np.full(num_sims, h_t_ns, dtype=np.float32),
            'AwayNetSpend': np.full(num_sims, a_t_ns, dtype=np.float32),
            'HomeNewManager': np.full(num_sims, h_new_mgr, dtype=np.float32),
            'AwayNewManager': np.full(num_sims, a_new_mgr, dtype=np.float32),
            'HomePromoted': np.full(num_sims, h_prom, dtype=np.float32),
            'AwayPromoted': np.full(num_sims, a_prom, dtype=np.float32),
        }
        X_sim = np.column_stack([col_map[c] for c in feature_names])
        xgb_probs = xgb_predictor.model.predict_proba(X_sim)

        lmbdas = np.clip(lmbda_base * (10.0 ** ((h_elos - baseline_elos[home]) / 400.0)), 1e-6, None)
        mus = np.clip(mu_base * (10.0 ** ((a_elos - baseline_elos[away]) / 400.0)), 1e-6, None)

        p_home = np.exp(-lmbdas[:, None] + np.arange(max_goals + 1)[None, :] * np.log(lmbdas[:, None]) - log_fact[None, :])
        p_away = np.exp(-mus[:, None] + np.arange(max_goals + 1)[None, :] * np.log(mus[:, None]) - log_fact[None, :])
        score_matrices = p_home[:, :, None] * p_away[:, None, :]

        adj_matrices = np.ones((num_sims, max_goals + 1, max_goals + 1))
        adj_matrices[:, 0, 0] = 1.0 - lmbdas * mus * dc_model.rho
        adj_matrices[:, 1, 0] = 1.0 + mus * dc_model.rho
        adj_matrices[:, 0, 1] = 1.0 + lmbdas * dc_model.rho
        adj_matrices[:, 1, 1] = 1.0 - dc_model.rho
        score_matrices *= np.clip(adj_matrices, 0.01, 10.0)
        score_matrices /= score_matrices.sum(axis=(1, 2))[:, None, None]

        dc_h = score_matrices[:, mask_h].sum(axis=1)
        dc_d = score_matrices[:, mask_d].sum(axis=1)
        dc_a = score_matrices[:, mask_a].sum(axis=1)

        blend_a = dc_weight * dc_a + (1.0 - dc_weight) * xgb_probs[:, 0]
        blend_d = dc_weight * dc_d + (1.0 - dc_weight) * xgb_probs[:, 1]
        blend_h = dc_weight * dc_h + (1.0 - dc_weight) * xgb_probs[:, 2]
        tot = blend_a + blend_d + blend_h
        blend_a /= tot
        blend_d /= tot
        blend_h /= tot

        score_matrices[:, mask_h] *= np.where(dc_h > 0, blend_h / dc_h, 0.0)[:, None]
        score_matrices[:, mask_d] *= np.where(dc_d > 0, blend_d / dc_d, 0.0)[:, None]
        score_matrices[:, mask_a] *= np.where(dc_a > 0, blend_a / dc_a, 0.0)[:, None]
        score_matrices /= score_matrices.sum(axis=(1, 2))[:, None, None]

        cum_probs = np.cumsum(score_matrices.reshape(num_sims, num_scores), axis=1)
        r = np.random.rand(num_sims, 1)
        goals = score_to_goals[(r > cum_probs).sum(axis=1)]
        hg = goals[:, 0]
        ag = goals[:, 1]

        h_pts = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        a_pts = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        points_sims[:, h_idx] += h_pts
        points_sims[:, a_idx] += a_pts
        gf_sims[:, h_idx] += hg
        gf_sims[:, a_idx] += ag
        gd_sims[:, h_idx] += hg - ag
        gd_sims[:, a_idx] += ag - hg

        # Update dynamic circular rolling buffers
        sim_h_sot = np.clip(1.4 + 1.4 * lmbdas + 0.6 * hg, 1.0, 12.0).astype(np.float32)
        sim_a_sot = np.clip(1.4 + 1.4 * mus + 0.6 * ag, 1.0, 12.0).astype(np.float32)
        sim_h_red = (np.random.rand(num_sims) < 0.06).astype(np.float32)
        sim_a_red = (np.random.rand(num_sims) < 0.06).astype(np.float32)

        hp_pos = int(roll_pos[h_idx])
        ap_pos = int(roll_pos[a_idx])
        roll_gf[:, h_idx, hp_pos] = hg.astype(np.float32)
        roll_ga[:, h_idx, hp_pos] = ag.astype(np.float32)
        roll_xg[:, h_idx, hp_pos] = lmbdas.astype(np.float32)
        roll_xga[:, h_idx, hp_pos] = mus.astype(np.float32)
        roll_sot_for[:, h_idx, hp_pos] = sim_h_sot
        roll_sot_against[:, h_idx, hp_pos] = sim_a_sot
        roll_red[:, h_idx, hp_pos] = sim_h_red
        roll_pts[:, h_idx, hp_pos] = h_pts.astype(np.float32)
        roll_pos[h_idx] = (hp_pos + 1) % ROLL_WIN
        roll_len[h_idx] = min(ROLL_WIN, int(roll_len[h_idx]) + 1)

        roll_gf[:, a_idx, ap_pos] = ag.astype(np.float32)
        roll_ga[:, a_idx, ap_pos] = hg.astype(np.float32)
        roll_xg[:, a_idx, ap_pos] = mus.astype(np.float32)
        roll_xga[:, a_idx, ap_pos] = lmbdas.astype(np.float32)
        roll_sot_for[:, a_idx, ap_pos] = sim_a_sot
        roll_sot_against[:, a_idx, ap_pos] = sim_h_sot
        roll_red[:, a_idx, ap_pos] = sim_a_red
        roll_pts[:, a_idx, ap_pos] = a_pts.astype(np.float32)
        roll_pos[a_idx] = (ap_pos + 1) % ROLL_WIN
        roll_len[a_idx] = min(ROLL_WIN, int(roll_len[a_idx]) + 1)

        # In-season Elo update across simulated outcomes
        e_h_sims = 1.0 / (1.0 + 10.0 ** ((a_elos - h_elos - 100.0) / 400.0))
        s_h_sims = np.where(hg > ag, 1.0, np.where(hg == ag, 0.5, 0.0))
        elo_sims[:, h_idx] = h_elos + 20.0 * (s_h_sims - e_h_sims)
        elo_sims[:, a_idx] = a_elos + 20.0 * ((1.0 - s_h_sims) - (1.0 - e_h_sims))

        if save_outputs:
            mean_score_matrix = score_matrices.mean(axis=0)
            max_idx = np.unravel_index(np.argmax(mean_score_matrix), mean_score_matrix.shape)
            preds_output.append({
                'MatchDay': matchday,
                'Date': date.strftime('%Y-%m-%d'),
                'HomeTeam': home,
                'AwayTeam': away,
                'Status': 'Scheduled',
                'ActualScore': None,
                'HomeWin%': round(float(blend_h.mean()) * 100, 2),
                'Draw%': round(float(blend_d.mean()) * 100, 2),
                'AwayWin%': round(float(blend_a.mean()) * 100, 2),
                'MostLikelyScore': f"{max_idx[0]}-{max_idx[1]}",
            })

    if save_outputs:
        preds_df = pd.DataFrame(preds_output)
        os.makedirs('output', exist_ok=True)
        preds_df.to_csv('output/predictions.csv', index=False)
        print("Saved predictions to output/predictions.csv")

    print("Simulation complete! Processing results...")
    for sim in range(num_sims):
        sort_indices = np.lexsort((gf_sims[sim, :], gd_sims[sim, :], points_sims[sim, :]))[::-1]
        ranks = np.zeros(num_teams, dtype=int)
        for rank_idx, team_idx in enumerate(sort_indices):
            ranks[team_idx] = rank_idx + 1
        ranks_sims[sim, :] = ranks

    summary_stats = []
    for i, team in enumerate(active_teams):
        team_points = points_sims[:, i]
        team_gds = gd_sims[:, i]
        team_rks = ranks_sims[:, i]
        summary_stats.append({
            'Team': team,
            'ExpectedPoints': round(np.median(team_points), 1),
            'PointsRange': f"{int(np.percentile(team_points, 5))}-{int(np.percentile(team_points, 95))}",
            'ExpectedGD': int(np.median(team_gds)),
            'GDRange': f"{int(np.percentile(team_gds, 5))}-{int(np.percentile(team_gds, 95))}",
            'ExpectedRank': int(np.median(team_rks)),
            'RankRange': f"{int(np.percentile(team_rks, 5))}-{int(np.percentile(team_rks, 95))}",
            'TitleWin%': round(np.mean(team_rks == 1) * 100, 2),
            'Top4%': round(np.mean(team_rks <= 4) * 100, 2),
            'Relegated%': round(np.mean(team_rks >= 18) * 100, 2),
        })

    summary_df = pd.DataFrame(summary_stats).sort_values('ExpectedRank').reset_index(drop=True)
    if save_outputs:
        os.makedirs('output', exist_ok=True)
        summary_df.to_csv('output/final_table_projection.csv', index=False)
        print("Saved standings projection to output/final_table_projection.csv")
    print("=== SEASON SIMULATION COMPLETED SUCCESSFULLY ===")

    return summary_df
