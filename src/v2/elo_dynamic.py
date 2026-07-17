import numpy as np

def get_expected_scores(h_elo, a_elo, hfa=100):
    """
    Calculate expected scores (win probabilities) using standard Elo logic.
    hfa: Home Field Advantage offset rating
    """
    e_h = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo - hfa) / 400.0))
    e_a = 1.0 - e_h
    return e_h, e_a

def get_gd_multiplier(gd):
    """
    Calculate goal difference index multiplier to scale Elo points:
    - If GD <= 1: multiplier = 1.0
    - If GD == 2: multiplier = 1.5
    - If GD == 3: multiplier = 1.75
    - If GD >= 4: multiplier = 1.75 + (GD - 3) / 8.0
    """
    abs_gd = abs(gd)
    if abs_gd <= 1:
        return 1.0
    elif abs_gd == 2:
        return 1.5
    elif abs_gd == 3:
        return 1.75
    else:
        return 1.75 + (abs_gd - 3) / 8.0

def update_elos(h_elo, a_elo, ftr, hg, ag, k_factor=20, hfa=100):
    """
    Compute updated Elo ratings for Home and Away team after a match.
    ftr: Final Time Result ('H', 'D', 'A')
    hg: Home goals scored
    ag: Away goals scored
    """
    e_h, e_a = get_expected_scores(h_elo, a_elo, hfa)
    
    # Map match outcome to scores
    if ftr == 'H':
        s_h, s_a = 1.0, 0.0
    elif ftr == 'A':
        s_h, s_a = 0.0, 1.0
    else:
        s_h, s_a = 0.5, 0.5
        
    gd = hg - ag
    mult = get_gd_multiplier(gd)
    
    # Calculate Elo updates
    h_elo_new = h_elo + k_factor * mult * (s_h - e_h)
    a_elo_new = a_elo + k_factor * mult * (s_a - e_a)
    
    return h_elo_new, a_elo_new
