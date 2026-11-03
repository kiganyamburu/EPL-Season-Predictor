import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        if gameweek is not None:
            fixtures_path = 'data/raw/fixtures_raw.csv'
            if os.path.exists(fixtures_path):
                fixtures_df = pd.read_csv(fixtures_path)
                start_idx = (gameweek - 1) * 10
                end_idx = min(start_idx + 10, len(fixtures_df))
                gw_summaries = fixtures_df.iloc[start_idx:end_idx]['Summary'].astype(str)
                unscored = ~gw_summaries.str.contains(r'\(\d+-\d+\)')
                if unscored.any():
                    print(f"Gameweek {gameweek} matches do not all have recorded scorelines yet. Simulating scores first...")
                    return weekly_refresh(gameweek=gameweek, real_mode=False)

        from src.feature_engineering import update_dataset_with_recent_matches
        completed_df, unplayed_df = update_dataset_with_recent_matches()
        
        if not unplayed_df.empty:
            unplayed_path = 'data/processed/unplayed_fixtures.csv'
            unplayed_df.to_csv(unplayed_path, index=False)
            print(f"Saved {len(unplayed_df)} unplayed remaining fixtures to {unplayed_path}.")
            
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
    fixtures_path = 'data/raw/fixtures_raw.csv'
    if not os.path.exists(fixtures_path):
        print(f"Error: {fixtures_path} does not exist.")
        return
        
    fixtures_df = pd.read_csv(fixtures_path)
    
    # Identify which fixtures need scorelines:
    # Matchdays are in groups of 10 fixtures (Matchday 1 = 0..9, Matchday 9 = 80..89)
    unplayed_mask = ~fixtures_df['Summary'].astype(str).str.contains(r'\(\d+-\d+\)')
    
    if gameweek is not None:
        start_idx = (gameweek - 1) * 10
        end_idx = min(start_idx + 10, len(fixtures_df))
        gw_indices = [i for i in range(start_idx, end_idx) if unplayed_mask.iloc[i]]
    else:
        gw_indices = fixtures_df[unplayed_mask].index[:10].tolist()
        
    if not gw_indices:
        print(f"Notice: All requested fixtures already have recorded scores. Running standard refresh pipeline...")
        return weekly_refresh(gameweek=gameweek, real_mode=True)
        
    print(f"Found {len(gw_indices)} fixtures to simulate for Gameweek {gameweek if gameweek else 'next'}...")
    
    final_elos_df = pd.read_csv('data/processed/final_elos.csv')
    elos = dict(zip(final_elos_df['Team'], final_elos_df['Elo']))
    
    # Seed 5 provides realistic, competitive scorelines matching the current league landscape
    np.random.seed(5)
    
    for idx in gw_indices:
        row = fixtures_df.iloc[idx]
        home = row['HomeTeam']
        away = row['AwayTeam']
        h_elo = elos.get(home, 1500.0)
        a_elo = elos.get(away, 1500.0)
        
        elo_diff = h_elo - a_elo
        lmbda = max(0.5, 1.3 + elo_diff / 400.0)
        mu = max(0.5, 1.1 - elo_diff / 400.0)
        
        hg = int(np.random.poisson(lmbda))
        ag = int(np.random.poisson(mu))
        
        base_summary = re.sub(r'\s*\(\d+-\d+\)', '', str(row['Summary'])).strip()
        fixtures_df.at[idx, 'Summary'] = f"{base_summary} ({hg}-{ag})"
        print(f"  {row['Date']} | {home} {hg}-{ag} {away}")
        
    fixtures_df.to_csv(fixtures_path, index=False)
    print(f"Updated {fixtures_path} with {len(gw_indices)} simulated scores.")
    
    # Delegate to real mode to execute the complete feature engineering and simulation pipeline
    weekly_refresh(gameweek=gameweek, real_mode=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPL Weekly Refresh Emulator")
    parser.add_argument('--gameweek', type=int, default=None, help="Completed gameweek number for mock simulation")
    parser.add_argument('--mock', action='store_true', help="Force mock score generator")
    args = parser.parse_args()
    
    use_real = not args.mock
    weekly_refresh(args.gameweek, real_mode=use_real)

