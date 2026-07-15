import os
import numpy as np
import pandas as pd
from scipy.stats import poisson

def run_season_simulations(fixtures_df, final_elos, df_historical, managers_path='data/raw/managers.csv', num_sims=10000, dc_weight=0.5):
    """
    Run Monte Carlo simulations of the 2026/27 EPL season.
    Uses a highly optimized vectorized discrete sampler for speed.
    """
    print(f"=== INITIALIZING MONTE CARLO SIMULATOR ({num_sims} RUNS) ===")
    
    # 1. Fit the Dixon-Coles model on the full historical dataset
    from src.models import DixonColesModel, XGBoostPredictor, blend_predictions
    
    dc_model = DixonColesModel()
    dc_model.fit(df_historical)
    
    # 2. Fit the XGBoost model on the historical dataset
    xgb_predictor = XGBoostPredictor()
    xgb_predictor.fit(df_historical)
    
    # 3. Load team metadata
    squad_df = pd.read_csv('data/raw/squad_values.csv')
    squad_values = dict(zip(squad_df['Team'], squad_df['MarketValue_M_Euros']))
    max_val = max(squad_values.values())
    min_val = min(squad_values.values())
    
    mgr_df = pd.read_csv(managers_path)
    mgr_df['AppointedDate'] = pd.to_datetime(mgr_df['AppointedDate'])
    managers = dict(zip(mgr_df['Team'], mgr_df['AppointedDate']))
    
    # Match dates logic to compute manager tenure
    fixtures_df['Date'] = pd.to_datetime(fixtures_df['Date'])
    
    # Compute features for 2026/27 fixtures
    # Get Elo for the 2026/27 season (regressed from 2025/26 final Elos)
    # The final_elos map contains the Elos at the end of 2025/26
    # Let's find promoted/relegated teams of 2026/27
    active_teams_2627 = set(fixtures_df['HomeTeam']).union(set(fixtures_df['AwayTeam']))
    teams_2526 = set(df_historical[df_historical['Season'] == '2025-2026']['HomeTeam']).union(
        set(df_historical[df_historical['Season'] == '2025-2026']['AwayTeam'])
    )
    
    relegated_2526 = teams_2526 - active_teams_2627
    promoted_2627 = active_teams_2627 - teams_2526
    
    # Regress Elos
    elo_2627 = {}
    if relegated_2526:
        relegated_elos = [final_elos.get(t, 1500.0) for t in relegated_2526]
        promoted_baseline = np.mean(relegated_elos)
    else:
        promoted_baseline = 1420.0
        
    for t in active_teams_2627:
        if t in final_elos and t not in promoted_2627:
            # Regress toward 1500
            elo_2627[t] = 0.7 * final_elos[t] + 0.3 * 1500.0
        else:
            elo_2627[t] = promoted_baseline
            
    # List of 20 teams in alphabetical order to make index lookup consistent
    teams_list = sorted(list(active_teams_2627))
    team_to_idx = {team: i for i, team in enumerate(teams_list)}
    
    # We will build a fixture list with model features
    fixtures_features = []
    
    # Dixon-Coles fixture probabilities and score parameters
    match_probabilities = []
    
    # We need to construct the score probability matrix (up to 10 goals) for all 380 fixtures
    # to feed into the vectorized simulation sampler
    max_goals = 10
    num_scores = (max_goals + 1) ** 2 # 121
    
    # Map score index to goals: score_idx = hg * 11 + ag
    score_to_goals = []
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            score_to_goals.append((hg, ag))
    score_to_goals = np.array(score_to_goals) # shape (121, 2)
    
    # Cumulative probability sum matrix for all matches
    # Shape: (380, 121)
    cum_probs_matrix = np.zeros((len(fixtures_df), num_scores))
    
    # Track probabilities for the predictions output file
    preds_output = []
    
    for idx, row in fixtures_df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        date = row['Date']
        
        # Calculate features for XGBoost
        h_elo = elo_2627[home]
        a_elo = elo_2627[away]
        
        # Manager tenure
        h_appointed = managers.get(home)
        h_tenure = (date - h_appointed).days if h_appointed else 365
        h_new_mgr = 1 if h_tenure < 180 else 0
        
        a_appointed = managers.get(away)
        a_tenure = (date - a_appointed).days if a_appointed else 365
        a_new_mgr = 1 if a_tenure < 180 else 0
        
        # Promoted
        h_promoted = 1 if home in promoted_2627 else 0
        a_promoted = 1 if away in promoted_2627 else 0
        
        # Squad value percentiles
        h_val = squad_values.get(home, 100.0)
        h_val_pct = (h_val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        
        a_val = squad_values.get(away, 100.0)
        a_val_pct = (a_val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        
        # Approximate rolling averages from historical values
        # (For simplicity and robustness, we seed rolling values with overall average strength coefficients)
        # Home attack/defense coefficients
        h_idx_dc = dc_model.team_indices.get(home)
        a_idx_dc = dc_model.team_indices.get(away)
        
        h_att = dc_model.attack_params[h_idx_dc] if h_idx_dc is not None else 1.0
        h_def = dc_model.defense_params[h_idx_dc] if h_idx_dc is not None else 1.0
        a_att = dc_model.attack_params[a_idx_dc] if a_idx_dc is not None else 1.0
        a_def = dc_model.defense_params[a_idx_dc] if a_idx_dc is not None else 1.0
        
        # Prepare feature dict for XGBoost
        feat_dict = {
            'EloDiff': h_elo - a_elo,
            'HomeRollingGoalsScored': h_att * 1.35, # scale by average goals
            'HomeRollingGoalsConceded': h_def * 1.15,
            'HomeRollingPoints': 1.6 if h_elo > 1550 else (1.1 if h_elo < 1450 else 1.3),
            'AwayRollingGoalsScored': a_att * 1.15,
            'AwayRollingGoalsConceded': a_def * 1.35,
            'AwayRollingPoints': 1.4 if a_elo > 1550 else (0.9 if a_elo < 1450 else 1.1),
            'HomeSquadValPercentile': h_val_pct,
            'AwaySquadValPercentile': a_val_pct,
            'HomeNewManager': h_new_mgr,
            'AwayNewManager': a_new_mgr,
            'HomePromoted': h_promoted,
            'AwayPromoted': a_promoted
        }
        
        # 1. Predictions from Dixon-Coles
        dc_preds = dc_model.predict_match_probabilities(home, away, max_goals=max_goals)
        
        # 2. Predictions from XGBoost
        # Create a single-row DataFrame for prediction
        x_df = pd.DataFrame([feat_dict])
        xgb_preds_prob = xgb_predictor.predict_probabilities(x_df)
        xgb_preds = {
            'HomeWin': xgb_preds_prob['HomeWin'][0],
            'Draw': xgb_preds_prob['Draw'][0],
            'AwayWin': xgb_preds_prob['AwayWin'][0]
        }
        
        # 3. Blend predictions
        blended = blend_predictions(dc_preds, xgb_preds, dc_weight=dc_weight)
        
        # To maintain the Dixon-Coles goal structure but align with the blended outcomes,
        # we adjust the score matrix.
        # Original Dixon-Coles score probabilities
        score_matrix = dc_preds['ScoreMatrix'].copy()
        
        # Sum of original outcomes
        sum_h = np.sum(np.triu(score_matrix, 1).T)
        sum_d = np.sum(np.diag(score_matrix))
        sum_a = np.sum(np.tril(score_matrix, -1).T)
        
        # Rescale score matrix quadrants to match the blended H/D/A probabilities
        for x in range(max_goals + 1):
            for y in range(max_goals + 1):
                if x > y:
                    score_matrix[x, y] *= blended['HomeWin'] / sum_h if sum_h > 0 else 0
                elif x < y:
                    score_matrix[x, y] *= blended['AwayWin'] / sum_a if sum_a > 0 else 0
                else:
                    score_matrix[x, y] *= blended['Draw'] / sum_d if sum_d > 0 else 0
                    
        # Re-normalize just in case
        score_matrix /= score_matrix.sum()
        
        # Flatten score matrix and compute cumulative sum for sampling
        flat_probs = score_matrix.flatten()
        cum_probs_matrix[idx, :] = np.cumsum(flat_probs)
        
        # Log match predictions
        preds_output.append({
            'MatchDay': idx // 10 + 1,
            'Date': date.strftime('%Y-%m-%d'),
            'HomeTeam': home,
            'AwayTeam': away,
            'HomeWin%': round(blended['HomeWin'] * 100, 2),
            'Draw%': round(blended['Draw'] * 100, 2),
            'AwayWin%': round(blended['AwayWin'] * 100, 2),
            'MostLikelyScore': dc_preds['MostLikelyScore']
        })
        
    # Write predictions to output/predictions.csv
    preds_df = pd.DataFrame(preds_output)
    os.makedirs('output', exist_ok=True)
    preds_df.to_csv('output/predictions.csv', index=False)
    print("Saved predictions to output/predictions.csv")
    
    # 4. SIMULATION LOOP (Vectorized)
    num_teams = len(teams_list)
    num_fixtures = len(fixtures_df)
    
    # Match team indices
    home_idx = fixtures_df['HomeTeam'].map(team_to_idx).values
    away_idx = fixtures_df['AwayTeam'].map(team_to_idx).values
    
    # We will accumulate team stats across all simulations
    # Shapes: (num_sims, num_teams)
    sim_points = np.zeros((num_sims, num_teams), dtype=int)
    sim_gd = np.zeros((num_sims, num_teams), dtype=int)
    sim_gf = np.zeros((num_sims, num_teams), dtype=int)
    sim_ranks = np.zeros((num_sims, num_teams), dtype=int)
    
    print("Running Monte Carlo simulation...")
    
    for sim in range(num_sims):
        # Generate random numbers for all 380 games
        r = np.random.rand(num_fixtures, 1)
        
        # Vectorized discrete sampling: find which bin each random float falls into
        # cum_probs_matrix is (380, 121)
        # (r > cum_probs_matrix).sum(axis=1) yields the index in [0, 120]
        score_indices = (r > cum_probs_matrix).sum(axis=1)
        
        # Map score index to goals scored
        # score_to_goals has shape (121, 2)
        goals = score_to_goals[score_indices] # shape (380, 2)
        hg = goals[:, 0]
        ag = goals[:, 1]
        
        # Calculate points for each game
        # 3 for win, 1 for draw, 0 for loss
        h_pts = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        a_pts = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        
        # Reset standings arrays for this simulation run
        points = np.zeros(num_teams, dtype=int)
        gd = np.zeros(num_teams, dtype=int)
        gf = np.zeros(num_teams, dtype=int)
        
        # Accumulate match statistics for each team using np.bincount
        # np.bincount adds values at indices: np.bincount(home_idx, weights=h_pts)
        points += np.bincount(home_idx, weights=h_pts, minlength=num_teams).astype(int)
        points += np.bincount(away_idx, weights=a_pts, minlength=num_teams).astype(int)
        
        gf += np.bincount(home_idx, weights=hg, minlength=num_teams).astype(int)
        gf += np.bincount(away_idx, weights=ag, minlength=num_teams).astype(int)
        
        # Goal differences
        gd += np.bincount(home_idx, weights=(hg - ag), minlength=num_teams).astype(int)
        gd += np.bincount(away_idx, weights=(ag - hg), minlength=num_teams).astype(int)
        
        # Determine Ranks
        # We want to sort teams descending by Points, then GD, then GF
        # To do this in numpy, we can create a composite score or use lexsort.
        # lexsort sorts ascending by the keys provided (keys are sorted in order from last to first)
        # So we want to sort ascending by: GF, GD, Points, and then flip to get descending.
        # Since lexsort sorts ascending, we use: gf, gd, points
        sort_indices = np.lexsort((gf, gd, points))[::-1]
        
        # Ranks: rank[sort_indices[0]] = 1, rank[sort_indices[1]] = 2, and so on.
        # The rank is 1-indexed (1 to 20)
        ranks = np.zeros(num_teams, dtype=int)
        for rank_idx, team_idx in enumerate(sort_indices):
            ranks[team_idx] = rank_idx + 1
            
        # Store simulation results
        sim_points[sim, :] = points
        sim_gd[sim, :] = gd
        sim_gf[sim, :] = gf
        sim_ranks[sim, :] = ranks
        
    print("Simulation complete! Processing results...")
    
    # 5. AGGREGATE RESULTS & COMPUTE PROBABILITIES
    summary_stats = []
    
    for i, team in enumerate(teams_list):
        team_points = sim_points[:, i]
        team_gds = sim_gd[:, i]
        team_rks = sim_ranks[:, i]
        
        title_pct = np.mean(team_rks == 1) * 100
        top4_pct = np.mean(team_rks <= 4) * 100
        relegated_pct = np.mean(team_rks >= 18) * 100
        
        # Calculate medians and 90% confidence intervals (5th to 95th percentile)
        med_pts = np.median(team_points)
        pts_5 = np.percentile(team_points, 5)
        pts_95 = np.percentile(team_points, 95)
        
        med_gd = np.median(team_gds)
        gd_5 = np.percentile(team_gds, 5)
        gd_95 = np.percentile(team_gds, 95)
        
        med_rank = np.median(team_rks)
        rk_5 = np.percentile(team_rks, 5)
        rk_95 = np.percentile(team_rks, 95)
        
        summary_stats.append({
            'Team': team,
            'ExpectedPoints': round(med_pts, 1),
            'PointsRange': f"{int(pts_5)}-{int(pts_95)}",
            'ExpectedGD': int(med_gd),
            'GDRange': f"{int(gd_5)}-{int(gd_95)}",
            'ExpectedRank': int(med_rank),
            'RankRange': f"{int(rk_5)}-{int(rk_95)}",
            'TitleWin%': round(title_pct, 2),
            'Top4%': round(top4_pct, 2),
            'Relegated%': round(relegated_pct, 2)
        })
        
    summary_df = pd.DataFrame(summary_stats)
    # Sort table by expected rank (ascending)
    summary_df = summary_df.sort_values('ExpectedRank').reset_index(drop=True)
    
    # Save output/final_table_projection.csv
    summary_df.to_csv('output/final_table_projection.csv', index=False)
    print("Saved standings projection to output/final_table_projection.csv")
    print("=== SEASON SIMULATION COMPLETED SUCCESSFULLY ===")
    
    return summary_df
