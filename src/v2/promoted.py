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

def blend_promoted_predictions(team, standard_lmbda, standard_mu, gameweek, is_promoted, is_home):
    """
    If a team is promoted, blend standard goals predictions with the historical baseline.
    Blend starts at 100% baseline (gameweek 0) and transitions to 100% standard by gameweek 10.
    """
    if not is_promoted:
        return standard_lmbda, standard_mu
        
    # Transition blending factor (max 10 games)
    baseline_weight = max(0.0, 1.0 - (gameweek / 10.0))
    standard_weight = 1.0 - baseline_weight
    
    # Baseline expectations
    # If the promoted team is home, their baseline lambda (scored) is BASELINE_PROMOTED_GF * home advantage,
    # and their baseline mu (conceded) is BASELINE_PROMOTED_GA / home advantage.
    # If away, it is the reverse.
    home_adv = 1.25
    if is_home:
        baseline_lmbda = BASELINE_PROMOTED_GF * home_adv
        baseline_mu = BASELINE_PROMOTED_GA
    else:
        baseline_lmbda = BASELINE_PROMOTED_GF
        baseline_mu = BASELINE_PROMOTED_GA * home_adv
        
    blended_lmbda = baseline_weight * baseline_lmbda + standard_weight * standard_lmbda
    blended_mu = baseline_weight * baseline_mu + standard_weight * standard_mu
    
    # Increase uncertainty (variance) for promoted teams during baseline period
    # We do this by slightly spreading their expected goals range or scaling parameter
    # (In Poisson model, variance = mean. To add extra variance/uncertainty, we add a tiny random noise
    # component to expected goals or scale the sampling range in simulation)
    if baseline_weight > 0:
        # Add a small dispersion factor
        dispersion = np.random.uniform(-0.1, 0.1) * baseline_weight
        blended_lmbda = np.maximum(0.1, blended_lmbda * (1.0 + dispersion))
        blended_mu = np.maximum(0.1, blended_mu * (1.0 - dispersion))
        
    return blended_lmbda, blended_mu
