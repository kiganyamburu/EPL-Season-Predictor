import os
import numpy as np
import pandas as pd

def calculate_elo_ratings(df, k_factor=20, hfa=100, regression_pct=0.3):
    """
    Calculate rolling Elo ratings for all teams chronologically.
    Regresses Elo ratings toward 1500 by regression_pct at season transitions.
    Assigns baseline Elo to promoted teams.
    """
    # Ensure chronological order
    df = df.sort_values('Date').reset_index(drop=True)
    
    elo = {}
    elo_history = []
    
    # Track the last season a team played
    current_season = None
    
    # Store season-end Elos to compute promoted team baselines
    season_final_elos = {}
    
    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        season = row['Season']
        
        # Check for season transition to apply regression
        if current_season is not None and season != current_season:
            # We have transitioned to a new season
            # Apply 30% regression to all active teams at the end of current_season
            active_teams_this_season = set(df[df['Season'] == current_season]['HomeTeam']).union(
                set(df[df['Season'] == current_season]['AwayTeam'])
            )
            
            # Save end-of-season Elos
            season_final_elos[current_season] = {t: elo.get(t, 1500) for t in active_teams_this_season}
            
            # Relegated teams are those that played in current_season but not in the new season
            teams_in_new_season = set(df[df['Season'] == season]['HomeTeam']).union(
                set(df[df['Season'] == season]['AwayTeam'])
            )
            
            relegated_teams = active_teams_this_season - teams_in_new_season
            promoted_teams = teams_in_new_season - active_teams_this_season
            
            # Determine baseline for promoted teams
            # Promoted baseline = average of relegated teams' Elo (or 1420 fallback if none)
            if relegated_teams:
                relegated_elos = [elo.get(t, 1500) for t in relegated_teams]
                promoted_baseline = np.mean(relegated_elos)
            else:
                promoted_baseline = 1420.0
                
            # Regress returning teams and assign promoted baselines
            for t in teams_in_new_season:
                if t in elo:
                    # Returning team: regress toward 1500
                    elo[t] = (1 - regression_pct) * elo[t] + regression_pct * 1500.0
                else:
                    # Promoted team: assign baseline Elo
                    elo[t] = promoted_baseline
                    
        current_season = season
        
        # Initialize Elo if team not seen yet
        if home not in elo:
            elo[home] = 1500.0
        if away not in elo:
            elo[away] = 1500.0
            
        # Record Elo before match
        h_elo = elo[home]
        a_elo = elo[away]
        
        elo_history.append({
            'HomeElo': h_elo,
            'AwayElo': a_elo,
            'EloDiff': h_elo - a_elo
        })
        
        # Calculate expected result
        # expected home win probability
        e_h = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo - hfa) / 400.0))
        e_a = 1.0 - e_h
        
        # Actual result
        if row['FTR'] == 'H':
            s_h, s_a = 1.0, 0.0
        elif row['FTR'] == 'A':
            s_h, s_a = 0.0, 1.0
        else:
            s_h, s_a = 0.5, 0.5
            
        # Update Elo
        elo[home] = h_elo + k_factor * (s_h - e_h)
        elo[away] = a_elo + k_factor * (s_a - e_a)
        
    elo_df = pd.DataFrame(elo_history)
    df = pd.concat([df, elo_df], axis=1)
    
    # Save final Elos at the end of the last historical season
    last_season = df['Season'].iloc[-1]
    active_teams_last_season = set(df[df['Season'] == last_season]['HomeTeam']).union(
        set(df[df['Season'] == last_season]['AwayTeam'])
    )
    season_final_elos[last_season] = {t: elo.get(t, 1500) for t in active_teams_last_season}
    
    return df, elo, season_final_elos

def calculate_rolling_features(df, rolling_window=10):
    """
    Calculate rolling points, form, goals scored, and goals conceded.
    """
    # Sort chronologically to ensure rolling calculation is correct
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Keep track of match history for each team
    team_history = {}
    
    rolling_stats = []
    
    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        
        # Initialize history if team not seen yet
        for team in (home, away):
            if team not in team_history:
                team_history[team] = []
                
        # Calculate features BEFORE the match
        h_stats = get_rolling_stats(team_history[home], rolling_window)
        a_stats = get_rolling_stats(team_history[away], rolling_window)
        
        rolling_stats.append({
            'HomeRollingGoalsScored': h_stats['goals_scored'],
            'HomeRollingGoalsConceded': h_stats['goals_conceded'],
            'HomeRollingPoints': h_stats['points'],
            'AwayRollingGoalsScored': a_stats['goals_scored'],
            'AwayRollingGoalsConceded': a_stats['goals_conceded'],
            'AwayRollingPoints': a_stats['points']
        })
        
        # Update match history AFTER the match
        # Home team update
        h_pts = 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)
        team_history[home].append({
            'goals_scored': row['FTHG'],
            'goals_conceded': row['FTAG'],
            'points': h_pts,
            'is_home': True
        })
        
        # Away team update
        a_pts = 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)
        team_history[away].append({
            'goals_scored': row['FTAG'],
            'goals_conceded': row['FTHG'],
            'points': a_pts,
            'is_home': False
        })
        
    rolling_df = pd.DataFrame(rolling_stats)
    df = pd.concat([df, rolling_df], axis=1)
    return df

def get_rolling_stats(history, window):
    """Helper to compute average statistics over rolling window."""
    if not history:
        return {'goals_scored': 1.3, 'goals_conceded': 1.3, 'points': 1.1} # league average defaults
        
    recent = history[-window:]
    avg_goals_scored = np.mean([x['goals_scored'] for x in recent])
    avg_goals_conceded = np.mean([x['goals_conceded'] for x in recent])
    avg_points = np.mean([x['points'] for x in recent])
    
    return {
        'goals_scored': avg_goals_scored,
        'goals_conceded': avg_goals_conceded,
        'points': avg_points
    }

def add_market_values_and_managers(df, squad_values_path='data/raw/squad_values.csv', managers_path='data/raw/managers.csv'):
    """
    Load and map squad values and manager tenure information.
    For historical data where exact squad values are not provided, we proxy
    market value rank using the team's standing from the previous season.
    """
    # Default values for historical matches if file not found
    squad_values = {}
    if os.path.exists(squad_values_path):
        squad_df = pd.read_csv(squad_values_path)
        squad_values = dict(zip(squad_df['Team'], squad_df['MarketValue_M_Euros']))
        
    # Calculate squad value percentiles for 2026/27 season
    max_val = max(squad_values.values()) if squad_values else 1000.0
    min_val = min(squad_values.values()) if squad_values else 50.0
    
    def get_squad_value_percentile(team, season):
        if season == "2026-2027":
            # Use real squad values
            val = squad_values.get(team, 100.0)
            return (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        else:
            # Proxy squad values rank based on previous season standing
            # Since we don't have historical standing, we estimate based on team's overall average goals/Elo in that season
            # Or simplified: if they are top-6, give high value; if promoted, low value.
            return 0.5 # default intermediate value for historical XGBoost training

    df['HomeSquadValPercentile'] = df.apply(lambda r: get_squad_value_percentile(r['HomeTeam'], r['Season']), axis=1)
    df['AwaySquadValPercentile'] = df.apply(lambda r: get_squad_value_percentile(r['AwayTeam'], r['Season']), axis=1)
    
    # Manager changes lookup
    managers = {}
    if os.path.exists(managers_path):
        mgr_df = pd.read_csv(managers_path)
        # Convert AppointedDate to datetime
        mgr_df['AppointedDate'] = pd.to_datetime(mgr_df['AppointedDate'])
        managers = dict(zip(mgr_df['Team'], mgr_df['AppointedDate']))
        
    # Mark if manager is new (appointed in 2026, tenure < 12 months)
    def is_new_manager(team, date_str, season):
        if season == "2026-2027":
            appointed = managers.get(team)
            if appointed:
                match_date = pd.to_datetime(date_str)
                tenure_days = (match_date - appointed).days
                # Less than 6 months (180 days) is considered new manager
                return 1 if tenure_days < 180 else 0
            return 1 # default to new manager if unknown
        return 0 # assume stable manager for historical training data (no high variance flag)
        
    # Transfer features lookup
    transfers_path = 'data/raw/transfers.csv'
    transfer_impacts = {}
    net_spends = {}
    if os.path.exists(transfers_path):
        t_df = pd.read_csv(transfers_path)
        for team, group in t_df.groupby('Team'):
            ins = group[group['TransferType'] == 'In']
            outs = group[group['TransferType'] == 'Out']
            net_spend = ins['Fee_M_Euros'].sum() - outs['Fee_M_Euros'].sum()
            impact_score = ins['Importance'].sum() - outs['Importance'].sum()
            net_spends[team] = net_spend
            transfer_impacts[team] = impact_score

    def get_transfer_impact(team, season):
        if season == "2026-2027":
            return transfer_impacts.get(team, 0.0)
        return 0.0

    def get_net_spend(team, season):
        if season == "2026-2027":
            return net_spends.get(team, 0.0)
        return 0.0

    df['HomeTransferImpact'] = df.apply(lambda r: get_transfer_impact(r['HomeTeam'], r['Season']), axis=1)
    df['AwayTransferImpact'] = df.apply(lambda r: get_transfer_impact(r['AwayTeam'], r['Season']), axis=1)
    df['HomeNetSpend'] = df.apply(lambda r: get_net_spend(r['HomeTeam'], r['Season']), axis=1)
    df['AwayNetSpend'] = df.apply(lambda r: get_net_spend(r['AwayTeam'], r['Season']), axis=1)

    df['HomeNewManager'] = df.apply(lambda r: is_new_manager(r['HomeTeam'], r['Date'], r['Season']), axis=1)
    df['AwayNewManager'] = df.apply(lambda r: is_new_manager(r['AwayTeam'], r['Date'], r['Season']), axis=1)
    
    return df

def identify_promoted_teams(df):
    """
    Identify promoted teams dynamically:
    A team is promoted in season S if they played in season S but did not play in season S-1.
    For the first season in the dataset (2015-2016), we use a hardcoded list of promoted teams.
    """
    seasons = sorted(df['Season'].unique())
    promoted_by_season = {}
    
    # First season (2015-2016) promoted teams: Bournemouth, Watford, Norwich
    promoted_by_season[seasons[0]] = {'Bournemouth', 'Watford', 'Norwich'}
    
    # Calculate for subsequent seasons
    for i in range(1, len(seasons)):
        prev_teams = set(df[df['Season'] == seasons[i-1]]['HomeTeam']).union(
            set(df[df['Season'] == seasons[i-1]]['AwayTeam'])
        )
        curr_teams = set(df[df['Season'] == seasons[i]]['HomeTeam']).union(
            set(df[df['Season'] == seasons[i]]['AwayTeam'])
        )
        # Promoted are in current but not in previous
        promoted = curr_teams - prev_teams
        promoted_by_season[seasons[i]] = promoted
        
    # Add promoted flags
    def check_promoted(team, season):
        if season in promoted_by_season:
            return 1 if team in promoted_by_season[season] else 0
        return 0
        
    df['HomePromoted'] = df.apply(lambda r: check_promoted(r['HomeTeam'], r['Season']), axis=1)
    df['AwayPromoted'] = df.apply(lambda r: check_promoted(r['AwayTeam'], r['Season']), axis=1)
    
    return df

def run_feature_engineering_pipeline():
    print("=== STARTING FEATURE ENGINEERING PIPELINE ===")
    raw_results_path = 'data/processed/historical_results.csv'
    if not os.path.exists(raw_results_path):
        print(f"Error: {raw_results_path} does not exist. Run data collection first.")
        return
        
    df = pd.read_csv(raw_results_path)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    
    # 1. Elo Ratings
    df, final_elos, season_final_elos = calculate_elo_ratings(df)
    
    # 2. Rolling Form & Goals
    df = calculate_rolling_features(df)
    
    # 3. Promoted status
    df = identify_promoted_teams(df)
    
    # 4. Squad Values and Managers
    df = add_market_values_and_managers(df)
    
    # Save processed training dataset
    os.makedirs('data/processed', exist_ok=True)
    processed_df_path = 'data/processed/matches_with_features.csv'
    df.to_csv(processed_df_path, index=False)
    print(f"Processed match dataset with features saved to {processed_df_path}. Matches: {len(df)}")
    
    # Save final Elo ratings to raw/final_elos.csv for season simulation carry-over
    final_elo_df = pd.DataFrame(list(final_elos.items()), columns=['Team', 'Elo'])
    final_elo_df.to_csv('data/processed/final_elos.csv', index=False)
    print("Saved final Elo ratings to data/processed/final_elos.csv")
    
    print("=== FEATURE ENGINEERING PIPELINE COMPLETED ===")

if __name__ == "__main__":
    run_feature_engineering_pipeline()
