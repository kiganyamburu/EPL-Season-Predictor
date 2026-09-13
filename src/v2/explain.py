import os
import pandas as pd
import numpy as np

def explain_match_prediction(
    home_team, away_team, date_str,
    home_elo, away_elo,
    home_squad_val, away_squad_val,
    home_avail, away_avail,
    home_congested, away_congested,
    home_win_p, draw_p, away_win_p,
    transfers_path='data/raw/transfers.csv'
):
    """
    Generate a human-readable explanation of the contributing factors for a match prediction.
    Includes squad transfers (key arrivals & departures).
    """
    factors = []
    
    # 1. Elo Strength
    elo_diff = home_elo - away_elo
    if abs(elo_diff) > 50:
        favored = home_team if elo_diff > 0 else away_team
        factors.append(f"• **Team Strength (Elo)**: {favored} is favored due to a higher Elo rating (differential of {abs(elo_diff):.0f} points).")
    else:
        factors.append(f"• **Team Strength (Elo)**: Matchup is evenly balanced on historical Elo (differential of {abs(elo_diff):.0f} points).")
        
    # 2. Squad Value
    if pd.notna(home_squad_val) and pd.notna(away_squad_val) and home_squad_val > 0 and away_squad_val > 0:
        val_diff = home_squad_val - away_squad_val
        if abs(val_diff) > 100:
            favored = home_team if val_diff > 0 else away_team
            factors.append(f"• **Squad Market Value**: {favored} has a squad value advantage of €{abs(val_diff):.1f}M.")

    # 3. Transfer Activity (Signings & Departures)
    if os.path.exists(transfers_path):
        try:
            t_df = pd.read_csv(transfers_path)
            h_ins = t_df[(t_df['Team'] == home_team) & (t_df['TransferType'] == 'In')].sort_values('Importance', ascending=False)
            a_ins = t_df[(t_df['Team'] == away_team) & (t_df['TransferType'] == 'In')].sort_values('Importance', ascending=False)
            h_outs = t_df[(t_df['Team'] == home_team) & (t_df['TransferType'] == 'Out')].sort_values('Importance', ascending=False)
            a_outs = t_df[(t_df['Team'] == away_team) & (t_df['TransferType'] == 'Out')].sort_values('Importance', ascending=False)

            t_notes = []
            if not h_ins.empty:
                t_notes.append(f"{home_team} boosted by signing {h_ins.iloc[0]['Player']}")
            if not a_ins.empty:
                t_notes.append(f"{away_team} boosted by signing {a_ins.iloc[0]['Player']}")
            if not h_outs.empty:
                t_notes.append(f"{home_team} adjusted after departure of {h_outs.iloc[0]['Player']}")
            if not a_outs.empty:
                t_notes.append(f"{away_team} adjusted after departure of {a_outs.iloc[0]['Player']}")

            if t_notes:
                factors.append(f"• **Transfer Activity**: {'; '.join(t_notes[:2])}.")
        except Exception:
            pass
            
    # 4. Squad Availability / Injuries
    if home_avail < 0.98 or away_avail < 0.98:
        if abs(home_avail - away_avail) > 0.03:
            healthier = home_team if home_avail > away_avail else away_team
            factors.append(f"• **Injuries & Availability**: {healthier} is healthier with {max(home_avail, away_avail)*100:.0f}% availability vs. {min(home_avail, away_avail)*100:.0f}% for the opponent.")
        else:
            factors.append(f"• **Injuries & Availability**: Both squads are dealing with minor availability losses (~{100 - min(home_avail, away_avail)*100:.0f}% out).")
            
    # 5. Rest days & Fatigue
    if home_congested or away_congested:
        if home_congested and not away_congested:
            factors.append(f"• **Fatigue**: {home_team} is on short rest (midweek match congestion) while {away_team} is fully rested.")
        elif away_congested and not home_congested:
            factors.append(f"• **Fatigue**: {away_team} is on short rest (midweek match congestion) while {home_team} is fully rested.")
        else:
            factors.append(f"• **Fatigue**: Both teams are facing congested schedules (midweek matches).")
            
    # 6. Home Field Advantage
    factors.append(f"• **Home Advantage**: {home_team} benefits from playing at home (statistically boosts goal-scoring rate by ~22%).")
    
    # Summary
    fav_team = home_team if home_win_p > away_win_p else away_team
    higher_p = max(home_win_p, away_win_p)
    
    if abs(home_win_p - away_win_p) < 0.05:
        summary_text = f"**Prediction Summary**: This matchup is highly contested with a close outcome probability. A Draw ({draw_p*100:.1f}%) is a very strong possibility."
    else:
        summary_text = f"**Prediction Summary**: {fav_team} is predicted to win this fixture with **{higher_p*100:.1f}%** confidence."
        
    explanation = "\n".join(factors) + "\n\n" + summary_text
    return explanation

