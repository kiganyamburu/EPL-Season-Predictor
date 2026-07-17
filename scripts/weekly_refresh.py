import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

def weekly_refresh(gameweek):
    """
    Weekly refresh emulator:
    1. Simulates that gameweek <GW> has completed.
    2. Moves fixtures in gameweek <GW> from fixtures_raw.csv to completed matches database, generating mock scorelines.
    3. Re-runs ELO updates and rolling features.
    4. Re-runs predictions for all remaining fixtures.
    5. Saves prediction snapshots in output/predictions_history.csv to track model calibration over time.
    """
    print(f"=== RUNNING WEEKLY REFRESH EMULATOR FOR GAMEWEEK {gameweek} ===")
    
    matches_path = 'data/processed/matches_with_features.csv'
    fixtures_path = 'data/raw/fixtures_raw.csv'
    history_path = 'output/predictions_history.csv'
    
    if not (os.path.exists(matches_path) and os.path.exists(fixtures_path)):
        print("Error: Required data files missing.")
        return
        
    # Read fixtures
    fixtures_df = pd.read_csv(fixtures_path)
    # Parse date
    fixtures_df['Date'] = pd.to_datetime(fixtures_df['Date'])
    
    # Identify matches in the selected gameweek
    gw_matches = fixtures_df[fixtures_df['MatchDay'] == gameweek]
    if gw_matches.empty:
        print(f"Error: Gameweek {gameweek} not found in fixtures.")
        return
        
    print(f"Found {len(gw_matches)} fixtures in Gameweek {gameweek}. Simulating scorelines...")
    
    # Generate mock scores based on team Elo strength differential
    completed_matches = pd.read_csv(matches_path)
    completed_matches['Date'] = pd.to_datetime(completed_matches['Date'], format='mixed', dayfirst=True)
    
    # Get final Elo ratings
    final_elos_df = pd.read_csv('data/processed/final_elos.csv')
    elos = dict(zip(final_elos_df['Team'], final_elos_df['Elo']))
    
    gw_results = []
    for idx, row in gw_matches.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        h_elo = elos.get(home, 1500.0)
        a_elo = elos.get(away, 1500.0)
        
        # Simple lambda/mu expectation from Elo difference
        elo_diff = h_elo - a_elo
        lmbda = max(0.5, 1.3 + elo_diff / 400.0)
        mu = max(0.5, 1.1 - elo_diff / 400.0)
        
        # Sample score
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
    
    # Append to completed matches
    new_completed = pd.concat([completed_matches, gw_results_df], ignore_index=True)
    new_completed.to_csv(matches_path, index=False)
    print(f"Appended {len(gw_results)} simulated results to completed matches database.")
    
    # Remove completed fixtures from remaining fixtures list
    remaining_fixtures = fixtures_df[fixtures_df['MatchDay'] > gameweek]
    remaining_fixtures.to_csv(fixtures_path, index=False)
    print(f"Removed Gameweek {gameweek} matches from fixtures list. {len(remaining_fixtures)} remaining.")
    
    # Snapshot historical predictions
    # Load current matchday predictions
    preds_path = 'output/predictions.csv'
    if os.path.exists(preds_path):
        current_preds = pd.read_csv(preds_path)
        current_preds['SnapshotDate'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_preds['CompletedGameweek'] = gameweek
        
        # Append to history file
        if os.path.exists(history_path):
            history_df = pd.read_csv(history_path)
            updated_history = pd.concat([history_df, current_preds], ignore_index=True)
        else:
            updated_history = current_preds
            
        # Ensure output folder exists
        os.makedirs('output', exist_ok=True)
        updated_history.to_csv(history_path, index=False)
        print(f"Saved predictions snapshot to {history_path} (Total snapshot records: {len(updated_history)}).")
        
    # Re-run pipeline to refresh standing projections with new completed matches
    print("Re-running predictive pipeline to update season simulations...")
    import subprocess
    subprocess.run(["python", "-m", "src.pipeline_v2"], check=True)
    print("=== WEEKLY REFRESH COMPLETED SUCCESSFULLY ===")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPL Weekly Refresh Emulator")
    parser.add_argument('--gameweek', type=int, default=1, help="Completed gameweek number")
    args = parser.parse_args()
    
    weekly_refresh(args.gameweek)
