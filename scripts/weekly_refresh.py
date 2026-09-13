import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

def weekly_refresh(gameweek=None, real_mode=True):
    """
    Weekly refresh pipeline:
    1. If real_mode is True: parses completed matches with actual scorelines from fixtures_raw.csv,
       updates historical results, recalculates Elo and rolling stats, trims fixtures_raw.csv,
       and re-runs season simulations.
    2. If real_mode is False: emulates completing a specific gameweek with mock scores.
    """
    print(f"=== RUNNING WEEKLY REFRESH (REAL MODE: {real_mode}) ===")
    
    if real_mode:
        from src.feature_engineering import update_dataset_with_recent_matches
        completed_df, unplayed_df = update_dataset_with_recent_matches()
        
        if not unplayed_df.empty:
            fixtures_path = 'data/raw/fixtures_raw.csv'
            unplayed_df.to_csv(fixtures_path, index=False)
            print(f"Updated {fixtures_path} with {len(unplayed_df)} unplayed remaining fixtures.")
            
        history_path = 'output/predictions_history.csv'
        preds_path = 'output/predictions.csv'
        if os.path.exists(preds_path):
            current_preds = pd.read_csv(preds_path)
            current_preds['SnapshotDate'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            current_preds['CompletedMatchesCount'] = len(completed_df)
            
            if os.path.exists(history_path):
                history_df = pd.read_csv(history_path)
                updated_history = pd.concat([history_df, current_preds], ignore_index=True)
            else:
                updated_history = current_preds
                
            os.makedirs('output', exist_ok=True)
            updated_history.to_csv(history_path, index=False)
            print(f"Saved predictions snapshot to {history_path}.")

        print("Re-running predictive pipeline to update season simulations...")
        from src.pipeline_v2 import run_v2_pipeline
        run_v2_pipeline()
        print("=== WEEKLY REFRESH COMPLETED SUCCESSFULLY ===")
        return

    # Mock mode fallback for gameweek emulation
    matches_path = 'data/processed/matches_with_features.csv'
    fixtures_path = 'data/raw/fixtures_raw.csv'
    history_path = 'output/predictions_history.csv'
    
    if not (os.path.exists(matches_path) and os.path.exists(fixtures_path)):
        print("Error: Required data files missing.")
        return
        
    fixtures_df = pd.read_csv(fixtures_path)
    fixtures_df['Date'] = pd.to_datetime(fixtures_df['Date'])
    
    gw_matches = fixtures_df[fixtures_df['MatchDay'] == gameweek] if 'MatchDay' in fixtures_df.columns else fixtures_df.head(10)
    if gw_matches.empty:
        print(f"Error: Gameweek {gameweek} not found in fixtures.")
        return
        
    print(f"Found {len(gw_matches)} fixtures to simulate for Gameweek {gameweek}...")
    
    completed_matches = pd.read_csv(matches_path)
    completed_matches['Date'] = pd.to_datetime(completed_matches['Date'], format='mixed', dayfirst=True)
    
    final_elos_df = pd.read_csv('data/processed/final_elos.csv')
    elos = dict(zip(final_elos_df['Team'], final_elos_df['Elo']))
    
    gw_results = []
    for idx, row in gw_matches.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        h_elo = elos.get(home, 1500.0)
        a_elo = elos.get(away, 1500.0)
        
        elo_diff = h_elo - a_elo
        lmbda = max(0.5, 1.3 + elo_diff / 400.0)
        mu = max(0.5, 1.1 - elo_diff / 400.0)
        
        hg = np.random.poisson(lmbda)
        ag = np.random.poisson(mu)
        ftr = 'H' if hg > ag else ('A' if ag > hg else 'D')
        
        match_result = {
            'Season': '2026-2027',
            'Date': row['Date'].strftime('%d/%m/%Y'),
            'HomeTeam': home,
            'AwayTeam': away,
            'FTHG': hg,
            'FTAG': ag,
            'FTR': ftr,
            'HomeElo': h_elo,
            'AwayElo': a_elo,
            'EloDiff': elo_diff,
            'HomeRollingGoalsScored': lmbda,
            'HomeRollingGoalsConceded': mu,
            'HomeRollingPoints': 3.0 if ftr == 'H' else (1.0 if ftr == 'D' else 0.0),
            'AwayRollingGoalsScored': mu,
            'AwayRollingGoalsConceded': lmbda,
            'AwayRollingPoints': 3.0 if ftr == 'A' else (1.0 if ftr == 'D' else 0.0),
            'HomePromoted': 0,
            'AwayPromoted': 0,
            'HomeSquadValPercentile': 0.5,
            'AwaySquadValPercentile': 0.5,
            'HomeNewManager': 0,
            'AwayNewManager': 0
        }
        gw_results.append(match_result)
        
    gw_results_df = pd.DataFrame(gw_results)
    
    new_completed = pd.concat([completed_matches, gw_results_df], ignore_index=True)
    new_completed.to_csv(matches_path, index=False)
    print(f"Appended {len(gw_results)} simulated results to completed matches database.")
    
    remaining_fixtures = fixtures_df.drop(gw_matches.index)
    remaining_fixtures.to_csv(fixtures_path, index=False)
    print(f"Removed simulated matches from fixtures list. {len(remaining_fixtures)} remaining.")
    
    from src.pipeline_v2 import run_v2_pipeline
    run_v2_pipeline()
    print("=== WEEKLY REFRESH COMPLETED SUCCESSFULLY ===")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPL Weekly Refresh Emulator")
    parser.add_argument('--gameweek', type=int, default=None, help="Completed gameweek number for mock simulation")
    parser.add_argument('--mock', action='store_true', help="Force mock score generator")
    args = parser.parse_args()
    
    use_real = not args.mock
    weekly_refresh(args.gameweek, real_mode=use_real)

