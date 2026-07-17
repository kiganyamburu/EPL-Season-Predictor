import os
import pandas as pd
import numpy as np

# English teams participating in European competitions for the 2026/27 season
TEAMS_IN_EUROPE = {'Arsenal', 'Man City', 'Liverpool', 'Aston Villa', 'Man United', 'Tottenham', 'Chelsea'}

# Midweek European match weeks for 2026/27 (dates are Monday of the matchweek)
EURO_MIDWEEK_WEEKS = [
    '2026-09-14', '2026-09-28', '2026-10-19', '2026-11-02', '2026-11-23', '2026-12-07',
    '2027-01-18', '2027-01-25', '2027-02-15', '2027-03-08', '2027-04-12', '2027-04-26', '2027-05-03'
]

# Domestic Cup midweek slots (Monday of the matchweek)
CUP_MIDWEEK_WEEKS = [
    '2026-09-21',  # Carabao Cup R3
    '2026-10-26',  # Carabao Cup R4
    '2026-12-14',  # Carabao Cup QF
    '2027-01-04',  # Carabao Cup SF Leg 1
    '2027-01-25',  # Carabao Cup SF Leg 2 / FA Cup replays
    '2027-02-22',  # FA Cup R5
    '2027-03-15'   # FA Cup QF replays
]

def calculate_rest_days(team, date_str, df_historical):
    """
    Calculate rest days since last Premier League match.
    Also overlays European/Cup midweek calendars.
    """
    dt = pd.to_datetime(date_str)
    
    # 1. Base rest days from Premier League schedule
    # Find all prior matches for this team
    prior_matches = df_historical[
        ((df_historical['HomeTeam'] == team) | (df_historical['AwayTeam'] == team)) & 
        (df_historical['Date'] < dt)
    ]
    
    if prior_matches.empty:
        # First game of the season, assume fully rested
        rest_days = 14
    else:
        last_match_date = prior_matches['Date'].max()
        # Convert to Timestamp if it is a string just in case
        if isinstance(last_match_date, str):
            last_match_date = pd.to_datetime(last_match_date)
        rest_days = (dt - last_match_date).days
        
    # 2. Check European/Cup Midweek Congestion
    # If rest_days is long (e.g. 7 days), they might have played a midweek Cup or European game
    if rest_days > 4:
        # Check if the preceding Tuesday-Thursday fell within a European or Domestic Cup midweek
        # We check if the Monday of that matchweek is in our midweek list
        match_week_monday = dt - pd.Timedelta(days=dt.weekday()) # Monday of current week
        monday_str = match_week_monday.strftime('%Y-%m-%d')
        
        # European congestion
        if team in TEAMS_IN_EUROPE and monday_str in EURO_MIDWEEK_WEEKS:
            rest_days = 3  # Congested: played Tuesday/Wednesday/Thursday
            
        # Domestic Cup congestion (applies to top teams who are usually in late stages)
        elif team in TEAMS_IN_EUROPE and monday_str in CUP_MIDWEEK_WEEKS:
            rest_days = 3  # Congested: played midweek cup tie
            
    return rest_days

def add_congestion_features(df):
    """
    Process matches DataFrame to add rest days and congestion flags.
    """
    df = df.sort_values('Date').reset_index(drop=True)
    home_rests = []
    away_rests = []
    
    for idx, row in df.iterrows():
        # During historical calculation, df contains all historical matches.
        # To prevent look-ahead bias, calculate_rest_days only looks at matches BEFORE current index
        h_rest = calculate_rest_days(row['HomeTeam'], row['Date'], df.iloc[:idx])
        a_rest = calculate_rest_days(row['AwayTeam'], row['Date'], df.iloc[:idx])
        home_rests.append(h_rest)
        away_rests.append(a_rest)
        
    df['HomeDaysSinceLast'] = home_rests
    df['AwayDaysSinceLast'] = away_rests
    
    df['HomeCongested'] = np.where(df['HomeDaysSinceLast'] <= 3, 1, 0)
    df['AwayCongested'] = np.where(df['AwayDaysSinceLast'] <= 3, 1, 0)
    
    return df
