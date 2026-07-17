# Configuration settings for EPL Season Predictor v2

# Toggles for modular upgrades (to evaluate ON/OFF independently)
ENABLE_TIME_DECAY = True
ENABLE_DYNAMIC_ELO = True
ENABLE_PROMOTED_HANDLING = True
ENABLE_INJURIES = True
ENABLE_CONGESTION = True
ENABLE_BAYESIAN_FORM = True

# Tuning parameters
DECAY_PHI = 0.003            # Exponential time-decay rate for historical matches
ELO_K_FACTOR = 20            # Elo updates sensitivity
ELO_HFA = 100                # Elo home field advantage rating offset
FORM_BOOST_RATE = 0.02       # Form change per win/loss inside correlated simulation
FORM_LIMIT_MAX = 1.2         # Max bound for performance form scaling
FORM_LIMIT_MIN = 0.8         # Min bound for performance form scaling
PROMOTED_TRANSITION_W = 10   # Gameweek limit to transition promoted team standard model

# Data file paths
MOCK_INJURY_PATH = 'data/raw/injuries_mock.csv'
HISTORICAL_MATCHES_PATH = 'data/processed/matches_with_features.csv'
FIXTURES_PATH = 'data/raw/fixtures_raw.csv'
SQUAD_VALUES_PATH = 'data/raw/squad_values.csv'
MANAGERS_PATH = 'data/raw/managers.csv'

# Set default v2 configuration mapping
def get_v2_config():
    return {
        'decay_phi': DECAY_PHI if ENABLE_TIME_DECAY else 0.0,
        'dynamic_elo': ENABLE_DYNAMIC_ELO,
        'promoted_handling': ENABLE_PROMOTED_HANDLING,
        'injuries': ENABLE_INJURIES,
        'congestion': ENABLE_CONGESTION,
        'bayesian_form': ENABLE_BAYESIAN_FORM,
        'form_boost': FORM_BOOST_RATE,
        'promoted_transition': PROMOTED_TRANSITION_W
    }
