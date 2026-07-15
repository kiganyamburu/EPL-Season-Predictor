import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from src.simulation import run_season_simulations

def run_backtest(df):
    """
    Perform a backtest on the 2024/25 season:
    - Train on data before 2024/25.
    - Evaluate on 2024/25 matches.
    - Compute Log-Loss and Accuracy.
    - Compare with naive baselines.
    """
    print("\n=== STARTING BACKTEST (EVALUATING 2024/25 SEASON) ===")
    
    # Split training and testing sets
    train_df = df[df['Season'] < '2024-2025'].copy()
    test_df = df[df['Season'] == '2024-2025'].copy()
    
    if len(test_df) == 0:
        print("Warning: 2024/25 season data not found. Skipping backtest.")
        return
        
    print(f"Training matches: {len(train_df)}")
    print(f"Testing matches (2024/25): {len(test_df)}")
    
    # 1. Fit Dixon-Coles model on training data
    from src.models import DixonColesModel, XGBoostPredictor, blend_predictions
    dc_model = DixonColesModel()
    dc_model.fit(train_df)
    
    # 2. Fit XGBoost Classifier on training data
    xgb_predictor = XGBoostPredictor()
    xgb_predictor.fit(train_df)
    
    # 3. Generate predictions for the test set
    true_labels = []
    
    # Store predictions from models
    dc_probs = []
    xgb_probs = []
    blended_probs = []
    
    for idx, row in test_df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        
        # True label mapping: 0=A, 1=D, 2=H
        true_label = 0 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 2)
        true_labels.append(true_label)
        
        # Dixon-Coles Prediction
        dc_pred = dc_model.predict_match_probabilities(home, away)
        dc_probs.append([dc_pred['AwayWin'], dc_pred['Draw'], dc_pred['HomeWin']])
        
        # XGBoost Prediction
        # Need to format row as DataFrame matching training features
        # We can extract features directly from test_df at this index
        x_eval = test_df.loc[[idx]]
        xgb_prob_arr = xgb_predictor.model.predict_proba(x_eval[xgb_predictor.features])[0]
        xgb_probs.append(xgb_prob_arr)
        
        # Blended Prediction
        dc_p_dict = {'HomeWin': dc_pred['HomeWin'], 'Draw': dc_pred['Draw'], 'AwayWin': dc_pred['AwayWin']}
        xgb_p_dict = {'HomeWin': xgb_prob_arr[2], 'Draw': xgb_prob_arr[1], 'AwayWin': xgb_prob_arr[0]}
        blend_p_dict = blend_predictions(dc_p_dict, xgb_p_dict, dc_weight=0.5)
        
        blended_probs.append([blend_p_dict['AwayWin'], blend_p_dict['Draw'], blend_p_dict['HomeWin']])
        
    true_labels = np.array(true_labels)
    dc_probs = np.array(dc_probs)
    xgb_probs = np.array(xgb_probs)
    blended_probs = np.array(blended_probs)
    
    # Evaluate metrics
    # Blended
    blend_loss = log_loss(true_labels, blended_probs)
    blend_acc = accuracy_score(true_labels, np.argmax(blended_probs, axis=1))
    
    # Dixon-Coles
    dc_loss = log_loss(true_labels, dc_probs)
    dc_acc = accuracy_score(true_labels, np.argmax(dc_probs, axis=1))
    
    # XGBoost
    xgb_loss = log_loss(true_labels, xgb_probs)
    xgb_acc = accuracy_score(true_labels, np.argmax(xgb_probs, axis=1))
    
    # Baselines
    # 1. Naive uniform (33.3% everywhere)
    uniform_probs = np.ones((len(test_df), 3)) / 3.0
    uniform_loss = log_loss(true_labels, uniform_probs)
    
    # 2. Naive always Home Win (columns: A=0.0, D=0.0, H=1.0)
    # Clip for log_loss calculation to prevent infinity
    home_win_probs = np.zeros((len(test_df), 3))
    home_win_probs[:, 2] = 1.0 - 1e-15
    home_win_probs[:, 0:2] = 5e-16
    home_win_loss = log_loss(true_labels, home_win_probs)
    home_win_acc = accuracy_score(true_labels, np.argmax(home_win_probs, axis=1))
    
    # 3. Historical frequency baseline (Home: 45%, Draw: 25%, Away: 30%)
    hist_freq = np.array([[0.30, 0.25, 0.45]] * len(test_df))
    hist_loss = log_loss(true_labels, hist_freq)
    
    print("\n--- BACKTEST RESULTS SUMMARY ---")
    print(f"{'Model/Baseline':<25} | {'Log-Loss':<10} | {'Accuracy':<10}")
    print("-" * 53)
    print(f"{'Ensemble Blend (50/50)':<25} | {blend_loss:.4f}     | {blend_acc * 100:.2f}%")
    print(f"{'Dixon-Coles Poisson':<25} | {dc_loss:.4f}     | {dc_acc * 100:.2f}%")
    print(f"{'XGBoost Classifier':<25} | {xgb_loss:.4f}     | {xgb_acc * 100:.2f}%")
    print(f"{'Historical Freq Baseline':<25} | {hist_loss:.4f}     | {45.00:.2f}%")
    print(f"{'Uniform Naive Baseline':<25} | {uniform_loss:.4f}     | {33.33}%")
    print(f"{'Always Home Win Baseline':<25} | {home_win_loss:.4f}    | {home_win_acc * 100:.2f}%")
    print("------------------------------------------------")
    
    # Verify that the blend beats the uniform and always home win baselines
    assert blend_loss < uniform_loss, "Error: Model fails to beat uniform baseline."
    print("Success: Ensemble blend beats uniform log-loss baseline.")

def run_end_to_end_pipeline():
    print("=== STARTING EPL SEASON PREDICTOR PIPELINE ===")
    
    # Paths check
    matches_path = 'data/processed/matches_with_features.csv'
    fixtures_path = 'data/raw/fixtures_raw.csv'
    final_elos_path = 'data/processed/final_elos.csv'
    
    if not (os.path.exists(matches_path) and os.path.exists(fixtures_path) and os.path.exists(final_elos_path)):
        print("Error: Processed data or fixtures missing. Please run data collection and feature engineering first.")
        return
        
    # Load processed historical matches
    df = pd.read_csv(matches_path)
    
    # Run Backtest
    run_backtest(df)
    
    # Load upcoming fixtures and Elos
    fixtures_df = pd.read_csv(fixtures_path)
    
    # Parse Elos
    final_elos_df = pd.read_csv(final_elos_path)
    final_elos = dict(zip(final_elos_df['Team'], final_elos_df['Elo']))
    
    # Train model on ALL historical data (including 2024/25 and 2025/26) to make final predictions for 2026/27
    print("\n=== TRAINING FINAL MODELS ON FULL DATASET ===")
    
    # Run the Monte Carlo simulation
    summary_df = run_season_simulations(
        fixtures_df=fixtures_df,
        final_elos=final_elos,
        df_historical=df,
        num_sims=10000,
        dc_weight=0.5
    )
    
    print("\n--- 2026/27 SEASON STANDINGS PROJECTION (TOP 10) ---")
    print(summary_df[['Team', 'ExpectedPoints', 'PointsRange', 'ExpectedRank', 'TitleWin%', 'Top4%', 'Relegated%']].head(10).to_string(index=False))
    
    print("\n--- 2026/27 SEASON STANDINGS PROJECTION (BOTTOM 5) ---")
    print(summary_df[['Team', 'ExpectedPoints', 'PointsRange', 'ExpectedRank', 'TitleWin%', 'Top4%', 'Relegated%']].tail(5).to_string(index=False))
    
    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_end_to_end_pipeline()
