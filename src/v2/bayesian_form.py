import numpy as np

def initialize_form(teams):
    """
    Initialize all teams with a form index of 1.0.
    """
    return {team: 1.0 for team in teams}

def update_form_state(form_dict, home_team, away_team, ftr, boost_rate=0.02):
    """
    Update team form states inside a simulated season run based on the match outcome.
    ftr: Final Time Result ('H', 'D', 'A')
    """
    h_form = form_dict[home_team]
    a_form = form_dict[away_team]
    
    if ftr == 'H':
        # Home Win: boost Home, decay Away
        h_form = min(1.20, h_form + boost_rate)
        a_form = max(0.80, a_form - boost_rate)
    elif ftr == 'A':
        # Away Win: boost Away, decay Home
        a_form = min(1.20, a_form + boost_rate)
        h_form = max(0.80, h_form - boost_rate)
    else:
        # Draw: regress both toward 1.0 (mean reversion of form)
        h_form = 0.95 * h_form + 0.05 * 1.0
        a_form = 0.95 * a_form + 0.05 * 1.0
        
    form_dict[home_team] = h_form
    form_dict[away_team] = a_form
    
    return form_dict

def adjust_goals_with_form(lmbda, mu, home_form, away_form):
    """
    Scale expected goals for a match using the teams' active form states:
    - High home form boosts home expected goals, low away form boosts home expected goals.
    """
    lmbda_adj = lmbda * (home_form / away_form)
    mu_adj = mu * (away_form / home_form)
    return max(0.01, lmbda_adj), max(0.01, mu_adj)
