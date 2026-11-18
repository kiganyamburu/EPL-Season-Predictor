import bisect
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

def _estimate_match_xg(row, side):
    """Return actual HxG/AxG when present, else calibrated shots-based xG estimate."""
    xg_col = 'HxG' if side == 'H' else 'AxG'
    if xg_col in row and pd.notna(row[xg_col]):
        return float(row[xg_col])
    s_col, st_col, g_col = ('HS', 'HST', 'FTHG') if side == 'H' else ('AS', 'AST', 'FTAG')
    if st_col in row and s_col in row and pd.notna(row[st_col]) and pd.notna(row[s_col]):
        hst = float(row[st_col])
        hs = float(row[s_col])
        return 0.25 * hst + 0.04 * max(0.0, hs - hst)
    return float(row[g_col])


def _safe_match_stat(row, col, default):
    """Return finite float match statistic from row, falling back to neutral default."""
    if col in row and pd.notna(row[col]):
        return float(row[col])
    return float(default)


def calculate_rolling_features(df, rolling_window=10):
    """
    Calculate pre-match rolling points, goals, expected goals (xG/xGA),
    shots on target (HST/AST), and red-card discipline (HR/AR).
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
            'HomeRollingXG': h_stats['xg'],
            'HomeRollingXGA': h_stats['xga'],
            'HomeRollingSOTFor': h_stats['sot_for'],
            'HomeRollingSOTAgainst': h_stats['sot_against'],
            'HomeRollingRedCards': h_stats['red_cards'],
            'AwayRollingGoalsScored': a_stats['goals_scored'],
            'AwayRollingGoalsConceded': a_stats['goals_conceded'],
            'AwayRollingPoints': a_stats['points'],
            'AwayRollingXG': a_stats['xg'],
            'AwayRollingXGA': a_stats['xga'],
            'AwayRollingSOTFor': a_stats['sot_for'],
            'AwayRollingSOTAgainst': a_stats['sot_against'],
            'AwayRollingRedCards': a_stats['red_cards'],
        })
        
        # Update match history AFTER the match
        h_xg = _estimate_match_xg(row, 'H')
        a_xg = _estimate_match_xg(row, 'A')
        h_sot = _safe_match_stat(row, 'HST', 4.4)
        a_sot = _safe_match_stat(row, 'AST', 4.4)
        h_red = _safe_match_stat(row, 'HR', 0.0)
        a_red = _safe_match_stat(row, 'AR', 0.0)

        h_pts = 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)
        team_history[home].append({
            'goals_scored': row['FTHG'],
            'goals_conceded': row['FTAG'],
            'xg': h_xg,
            'xga': a_xg,
            'sot_for': h_sot,
            'sot_against': a_sot,
            'red_cards': h_red,
            'points': h_pts,
            'is_home': True
        })
        
        a_pts = 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)
        team_history[away].append({
            'goals_scored': row['FTAG'],
            'goals_conceded': row['FTHG'],
            'xg': a_xg,
            'xga': h_xg,
            'sot_for': a_sot,
            'sot_against': h_sot,
            'red_cards': a_red,
            'points': a_pts,
            'is_home': False
        })
        
    rolling_df = pd.DataFrame(rolling_stats)
    df = pd.concat([df, rolling_df], axis=1)
    return df

def get_rolling_stats(history, window):
    """Helper to compute average statistics over rolling window."""
    if not history:
        return {
            'goals_scored': 1.3,
            'goals_conceded': 1.3,
            'xg': 1.3,
            'xga': 1.3,
            'sot_for': 4.4,
            'sot_against': 4.4,
            'red_cards': 0.06,
            'points': 1.1,
        }
        
    recent = history[-window:]
    return {
        'goals_scored': float(np.mean([x['goals_scored'] for x in recent])),
        'goals_conceded': float(np.mean([x['goals_conceded'] for x in recent])),
        'xg': float(np.mean([x['xg'] for x in recent])),
        'xga': float(np.mean([x['xga'] for x in recent])),
        'sot_for': float(np.mean([x['sot_for'] for x in recent])),
        'sot_against': float(np.mean([x['sot_against'] for x in recent])),
        'red_cards': float(np.mean([x['red_cards'] for x in recent])),
        'points': float(np.mean([x['points'] for x in recent])),
    }

def _compute_prior_season_percentiles(df):
    """
    Build zero-leakage (season, team) -> [0, 1] strength/squad-value percentiles
    using each club's final points from season S-1 (promoted clubs map near 0.05).
    """
    seasons = sorted(df['Season'].unique())
    season_points = {}
    for s in seasons:
        s_df = df[df['Season'] == s]
        pts = {}
        for _, row in s_df.iterrows():
            h, a, ftr = row['HomeTeam'], row['AwayTeam'], row['FTR']
            pts[h] = pts.get(h, 0) + (3 if ftr == 'H' else (1 if ftr == 'D' else 0))
            pts[a] = pts.get(a, 0) + (3 if ftr == 'A' else (1 if ftr == 'D' else 0))
        season_points[s] = pts

    pct_map = {}
    for idx, s in enumerate(seasons):
        s_df = df[df['Season'] == s]
        teams_s = set(s_df['HomeTeam']).union(set(s_df['AwayTeam']))
        if idx == 0:
            for t in teams_s:
                pct_map[(s, t)] = 0.5
            continue

        prev_pts = season_points[seasons[idx - 1]]
        returning_vals = [prev_pts[t] for t in teams_s if t in prev_pts]
        if not returning_vals:
            for t in teams_s:
                pct_map[(s, t)] = 0.5
            continue

        min_ret = min(returning_vals)
        max_ret = max(returning_vals)
        promoted_proxy = max(0.0, min_ret - 5.0)
        denom = max_ret - promoted_proxy
        for t in teams_s:
            val = prev_pts.get(t, promoted_proxy)
            pct_map[(s, t)] = (val - promoted_proxy) / denom if denom > 0 else 0.5
    return pct_map


def _get_promoted_teams_by_season(df):
    """Map each season to the set of clubs newly promoted into that season."""
    seasons = sorted(df['Season'].unique())
    if not seasons:
        return {}
    promoted_by_season = {seasons[0]: {'Bournemouth', 'Watford', 'Norwich'}}
    for i in range(1, len(seasons)):
        prev_teams = set(df[df['Season'] == seasons[i - 1]]['HomeTeam']).union(
            set(df[df['Season'] == seasons[i - 1]]['AwayTeam'])
        )
        curr_teams = set(df[df['Season'] == seasons[i]]['HomeTeam']).union(
            set(df[df['Season'] == seasons[i]]['AwayTeam'])
        )
        promoted_by_season[seasons[i]] = curr_teams - prev_teams
    return promoted_by_season


def _load_manager_spell_lookup(managers_path, historical_managers_path):
    """Load sorted managerial appointment timestamps per club across historical and current files."""
    frames = []
    for path in (historical_managers_path, managers_path):
        if path and os.path.exists(path):
            m_df = pd.read_csv(path)
            if {'Team', 'AppointedDate'}.issubset(m_df.columns):
                frames.append(m_df[['Team', 'AppointedDate']])
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True)
    combined['AppointedDate'] = pd.to_datetime(combined['AppointedDate'], errors='coerce')
    combined = combined.dropna(subset=['Team', 'AppointedDate'])
    combined = combined.drop_duplicates(subset=['Team', 'AppointedDate'], keep='last')
    spells = {}
    for team, grp in combined.groupby('Team'):
        spells[team] = sorted(grp['AppointedDate'].tolist())
    return spells


def add_market_values_and_managers(
    df,
    squad_values_path='data/raw/squad_values.csv',
    managers_path='data/raw/managers.csv',
    historical_managers_path='data/raw/historical_managers.csv',
):
    """
    Load and map squad values, historical + current manager spells, and transfer metrics.
    For historical seasons without direct 2026/27 metadata, uses zero-leakage prior-season
    proxies on matching scales so tree models learn genuine historical signal.
    """
    squad_values = {}
    if os.path.exists(squad_values_path):
        squad_df = pd.read_csv(squad_values_path)
        squad_values = dict(zip(squad_df['Team'], squad_df['MarketValue_M_Euros']))

    max_val = max(squad_values.values()) if squad_values else 1000.0
    min_val = min(squad_values.values()) if squad_values else 50.0
    hist_pct_map = _compute_prior_season_percentiles(df)
    promoted_by_season = _get_promoted_teams_by_season(df)

    def get_squad_value_percentile(team, season):
        if season == "2026-2027" and team in squad_values:
            val = squad_values[team]
            return (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        return hist_pct_map.get((season, team), 0.5)

    df['HomeSquadValPercentile'] = [get_squad_value_percentile(t, s) for t, s in zip(df['HomeTeam'], df['Season'])]
    df['AwaySquadValPercentile'] = [get_squad_value_percentile(t, s) for t, s in zip(df['AwayTeam'], df['Season'])]

    # Multi-spell manager lookup across 2015-2027 appointments
    manager_spells = _load_manager_spell_lookup(managers_path, historical_managers_path)
    match_dates = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)

    def is_new_manager(team, match_date, season):
        spells = manager_spells.get(team)
        if spells:
            pos = bisect.bisect_right(spells, match_date) - 1
            if pos >= 0:
                tenure_days = (match_date - spells[pos]).days
                return 1 if 0 <= tenure_days < 180 else 0
            if season == "2026-2027":
                return 1
            return 0
        return 1 if season == "2026-2027" else 0

    df['HomeNewManager'] = [is_new_manager(t, d, s) for t, d, s in zip(df['HomeTeam'], match_dates, df['Season'])]
    df['AwayNewManager'] = [is_new_manager(t, d, s) for t, d, s in zip(df['AwayTeam'], match_dates, df['Season'])]

    # Transfer features lookup (exact 2026/27 records + zero-leakage prior-season proxies historically)
    transfers_path = 'data/raw/transfers.csv'
    transfer_impacts = {}
    net_spends = {}
    if os.path.exists(transfers_path):
        t_df = pd.read_csv(transfers_path)
        has_season_col = 'Season' in t_df.columns
        group_cols = ['Season', 'Team'] if has_season_col else ['Team']
        for key, group in t_df.groupby(group_cols):
            ins = group[group['TransferType'] == 'In']
            outs = group[group['TransferType'] == 'Out']
            ns_val = float(ins['Fee_M_Euros'].sum() - outs['Fee_M_Euros'].sum())
            imp_val = float(ins['Importance'].sum() - outs['Importance'].sum())
            season_key, team_key = key if has_season_col else ("2026-2027", key)
            net_spends[(season_key, team_key)] = ns_val
            transfer_impacts[(season_key, team_key)] = imp_val

    def get_transfer_features(team, season):
        if (season, team) in transfer_impacts:
            return transfer_impacts[(season, team)], net_spends[(season, team)]
        p = hist_pct_map.get((season, team), 0.5)
        is_prom = 1.0 if team in promoted_by_season.get(season, set()) else 0.0
        # Zero-leakage prior-season proxy matched in scale to transfers.csv
        impact_proxy = round(1.35 - 0.70 * p + 0.35 * is_prom, 4)
        net_spend_proxy = round(75.0 + 210.0 * p + 40.0 * is_prom, 2)
        return impact_proxy, net_spend_proxy

    h_transfers = [get_transfer_features(t, s) for t, s in zip(df['HomeTeam'], df['Season'])]
    a_transfers = [get_transfer_features(t, s) for t, s in zip(df['AwayTeam'], df['Season'])]
    df['HomeTransferImpact'] = [x[0] for x in h_transfers]
    df['AwayTransferImpact'] = [x[0] for x in a_transfers]
    df['HomeNetSpend'] = [x[1] for x in h_transfers]
    df['AwayNetSpend'] = [x[1] for x in a_transfers]

    return df

def identify_promoted_teams(df):
    """
    Identify promoted teams dynamically:
    A team is promoted in season S if they played in season S but did not play in season S-1.
    For the first season in the dataset (2015-2016), we use a hardcoded list of promoted teams.
    """
    promoted_by_season = _get_promoted_teams_by_season(df)

    def check_promoted(team, season):
        if season in promoted_by_season:
            return 1 if team in promoted_by_season[season] else 0
        return 0
        
    df['HomePromoted'] = df.apply(lambda r: check_promoted(r['HomeTeam'], r['Season']), axis=1)
    df['AwayPromoted'] = df.apply(lambda r: check_promoted(r['AwayTeam'], r['Season']), axis=1)
    
    return df

def update_dataset_with_recent_matches():
    """
    Check for completed match scorelines in fixtures_raw.csv, append them to historical results,
    and re-run complete feature engineering (Elo, rolling stats, squad value, managers).
    Returns (completed_df, unplayed_df).
    """
    from src.data_collection import parse_completed_fixtures_from_raw

    raw_results_path = 'data/processed/historical_results.csv'
    if not os.path.exists(raw_results_path):
        print(f"Error: {raw_results_path} does not exist.")
        return pd.DataFrame(), pd.DataFrame()

    df_hist = pd.read_csv(raw_results_path)
    
    # Parse completed matches from raw fixtures
    completed_2627, unplayed_2627 = parse_completed_fixtures_from_raw('data/raw')
    
    if not completed_2627.empty:
        # Ensure 2026/27 matches in historical results match current fixtures_raw.csv scorelines
        df_hist['MatchID'] = df_hist['Season'].astype(str) + '_' + df_hist['HomeTeam'].astype(str) + '_' + df_hist['AwayTeam'].astype(str)
        completed_2627['MatchID'] = completed_2627['Season'].astype(str) + '_' + completed_2627['HomeTeam'].astype(str) + '_' + completed_2627['AwayTeam'].astype(str)
        
        updated_ids = set(completed_2627['MatchID'])
        df_hist = df_hist[~df_hist['MatchID'].isin(updated_ids)].copy()
        
        df_hist = df_hist.drop(columns=['MatchID'])
        completed_clean = completed_2627.drop(columns=['MatchID'])
        
        print(f"Syncing {len(completed_clean)} completed 2026/27 matches to historical results...")
        combined_df = pd.concat([df_hist, completed_clean], ignore_index=True)
        # Update historical_results.csv
        combined_df.to_csv(raw_results_path, index=False)
    else:
        combined_df = df_hist

    combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='mixed', dayfirst=True)
    
    # Re-run full feature engineering pipeline
    combined_df, final_elos, season_final_elos = calculate_elo_ratings(combined_df)
    combined_df = calculate_rolling_features(combined_df)
    combined_df = identify_promoted_teams(combined_df)
    combined_df = add_market_values_and_managers(combined_df)
    
    # Save processed dataset
    os.makedirs('data/processed', exist_ok=True)
    processed_df_path = 'data/processed/matches_with_features.csv'
    combined_df.to_csv(processed_df_path, index=False)
    print(f"Updated matches_with_features.csv with {len(combined_df)} total matches.")
    
    # Save final Elos
    final_elo_df = pd.DataFrame(list(final_elos.items()), columns=['Team', 'Elo'])
    final_elo_df.to_csv('data/processed/final_elos.csv', index=False)
    print("Saved updated final Elo ratings to data/processed/final_elos.csv")
    
    return completed_2627, unplayed_2627

def run_feature_engineering_pipeline():
    print("=== STARTING FEATURE ENGINEERING PIPELINE ===")
    update_dataset_with_recent_matches()
    print("=== FEATURE ENGINEERING PIPELINE COMPLETED ===")

if __name__ == "__main__":
    run_feature_engineering_pipeline()

