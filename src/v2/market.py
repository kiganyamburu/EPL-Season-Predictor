import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

def de_vig_odds(h_odds, d_odds, a_odds):
    """
    Remove the bookmaker's overround (vigorish) by basic normalization.
    Returns implied probabilities for Home Win, Draw, and Away Win.
    """
    if pd.isna(h_odds) or pd.isna(d_odds) or pd.isna(a_odds) or h_odds <= 0 or d_odds <= 0 or a_odds <= 0:
        # Fallback to uniform probabilities if odds are invalid/missing
        return 1/3., 1/3., 1/3.
        
    implied_h = 1.0 / h_odds
    implied_d = 1.0 / d_odds
    implied_a = 1.0 / a_odds
    
    overround = implied_h + implied_d + implied_a
    
    p_h = implied_h / overround
    p_d = implied_d / overround
    p_a = implied_a / overround
    
    return p_h, p_d, p_a

def extract_market_probs(df):
    """
    Extract and de-vig market probabilities for a matches DataFrame.
    """
    market_probs = []
    
    # Try different bookmakers in order of preference
    bookmakers = [
        ('B365H', 'B365D', 'B365A'),
        ('PSH', 'PSD', 'PSA'),
        ('BWH', 'BWD', 'BWA'),
        ('IWH', 'IWD', 'IWA'),
        ('LBH', 'LBD', 'LBA')
    ]
    
    for idx, row in df.iterrows():
        p_h, p_d, p_a = 1/3., 1/3., 1/3.
        for h_col, d_col, a_col in bookmakers:
            if h_col in row and d_col in row and a_col in row:
                if not (pd.isna(row[h_col]) or pd.isna(row[d_col]) or pd.isna(row[a_col])):
                    p_h, p_d, p_a = de_vig_odds(row[h_col], row[d_col], row[a_col])
                    break
        market_probs.append([p_a, p_d, p_h])  # Align classes mapping: 0=A, 1=D, 2=H
        
    return np.array(market_probs)

def evaluate_market_benchmark(true_labels, model_probs, test_df):
    """
    Compare model's log-loss and Brier score vs. the betting market implied probabilities.
    true_labels: array of ints (0=A, 1=D, 2=H)
    model_probs: array of shape (N, 3) (columns: A, D, H)
    test_df: original matches DataFrame
    """
    # 1. Parse betting market probabilities
    market_probs = extract_market_probs(test_df)
    
    # Calculate Log-Loss
    model_loss = log_loss(true_labels, model_probs)
    market_loss = log_loss(true_labels, market_probs)
    
    # Calculate Brier score (Brier score is computed per class and averaged)
    # Convert true labels to one-hot encoding
    N = len(true_labels)
    true_one_hot = np.zeros((N, 3))
    true_one_hot[np.arange(N), true_labels] = 1.0
    
    model_brier = np.mean(np.sum((model_probs - true_one_hot)**2, axis=1)) / 3.0
    market_brier = np.mean(np.sum((market_probs - true_one_hot)**2, axis=1)) / 3.0
    
    # 2. Analyze disagreements
    # Disagreement: Euclidean distance between model & market probability distributions
    disagreements = np.linalg.norm(model_probs - market_probs, axis=1)
    
    # Model error: Euclidean distance between model predictions and true outcome
    model_errors = np.linalg.norm(model_probs - true_one_hot, axis=1)
    
    # Correlation coefficient between disagreement and error
    corr = np.corrcoef(disagreements, model_errors)[0, 1]
    
    print("\n--- BETTING MARKET BENCHMARK ---")
    print(f"Model Log-Loss:  {model_loss:.4f}  |  Market Log-Loss:  {market_loss:.4f}")
    print(f"Model Brier:     {model_brier:.4f}  |  Market Brier:     {market_brier:.4f}")
    print(f"Disagreement vs. Prediction Error Correlation: {corr:.4f}")
    
    return {
        'model_loss': model_loss,
        'market_loss': market_loss,
        'model_brier': model_brier,
        'market_brier': market_brier,
        'disagreement_error_correlation': corr,
        'market_probs': market_probs
    }
