import os
import pandas as pd
import numpy as np

# Average historical performance of promoted teams over last 10 seasons
BASELINE_PROMOTED_GF = 1.00  # Goals scored per match
BASELINE_PROMOTED_GA = 1.70  # Goals conceded per match

def get_promoted_teams(df_historical, season):
    """
    Identify promoted teams for a given season:
    Played in 'season' but not in 'season-1'.
    """
    seasons = sorted(df_historical['Season'].unique())
    if season not in seasons:
        # If it's the upcoming season (e.g. 2026-2027) which is not in the historical DF yet
        # We can find all teams in the fixtures but not in the final historical season
        return {'Coventry', 'Ipswich', 'Hull'}
        
    idx = seasons.index(season)
    if idx == 0:
        return {'Bournemouth', 'Watford', 'Norwich'}  # Fallback for first season
        
    prev_teams = set(df_historical[df_historical['Season'] == seasons[idx-1]]['HomeTeam']).union(
        set(df_historical[df_historical['Season'] == seasons[idx-1]]['AwayTeam'])
    )
    curr_teams = set(df_historical[df_historical['Season'] == seasons[idx]]['HomeTeam']).union(
        set(df_historical[df_historical['Season'] == seasons[idx]]['AwayTeam'])
    )
    
    return curr_teams - prev_teams

def blend_promoted_predictions(team, standard_lmbda, standard_mu, gameweek, is_promoted, is_home, add_dispersion=False):
    """
    If a team is promoted, blend standard goals predictions with the historical baseline.
    Blend starts at baseline weight at gameweek 0 and transitions smoothly to standard by gameweek 10.
    """
    if not is_promoted:
        return standard_lmbda, standard_mu
        
    # Transition blending factor (smooth cosine/linear over 10 games)
    baseline_weight = max(0.0, 1.0 - (gameweek / 10.0)) * 0.60  # Scale maximum baseline anchor to 60% to avoid crushing opponent attack
    standard_weight = 1.0 - baseline_weight
    
    home_adv = 1.20
    if is_home:
        baseline_lmbda = BASELINE_PROMOTED_GF * home_adv
        baseline_mu = np.maximum(standard_mu, BASELINE_PROMOTED_GA * 0.90)
    else:
        baseline_lmbda = BASELINE_PROMOTED_GF
        baseline_mu = np.maximum(standard_mu, BASELINE_PROMOTED_GA * home_adv * 0.90)
        
    blended_lmbda = baseline_weight * baseline_lmbda + standard_weight * standard_lmbda
    blended_mu = baseline_weight * baseline_mu + standard_weight * standard_mu
    
    if add_dispersion and baseline_weight > 0:
        dispersion = np.random.uniform(-0.05, 0.05) * baseline_weight
        blended_lmbda = np.maximum(0.1, blended_lmbda * (1.0 + dispersion))
        blended_mu = np.maximum(0.1, blended_mu * (1.0 - dispersion))
        
    return blended_lmbda, blended_mu

