# Configuration settings for EPL Season Predictor v2
import json
import os

# Toggles for modular upgrades (to evaluate ON/OFF independently)
ENABLE_TIME_DECAY = True
ENABLE_DYNAMIC_ELO = True
ENABLE_PROMOTED_HANDLING = True
ENABLE_INJURIES = True
ENABLE_CONGESTION = True
ENABLE_BAYESIAN_FORM = True
ENABLE_DYNAMIC_CALIBRATION = True
ENABLE_DC_XG_BLEND = True

# Tuning parameters (calibrated via walk-forward CV)
DECAY_PHI = 0.0012            # Exponential time-decay rate (half-life ~1.58 years)
DC_WEIGHT = 0.35              # Optimal blend weight for Dixon-Coles vs XGBoost
DC_XG_WEIGHT = 0.40           # Blend weight for HxG/AxG in Dixon-Coles rate estimation
CALIBRATION_TEMP = 1.12       # Fallback out-of-fold temperature calibration smoothing factor
ELO_K_FACTOR = 20             # Elo updates sensitivity
ELO_HFA = 100                 # Elo home field advantage rating offset
FORM_BOOST_RATE = 0.01        # Calibrated form change per win/loss inside correlated simulation
FORM_LIMIT_MAX = 1.10         # Max bound for performance form scaling
FORM_LIMIT_MIN = 0.90         # Min bound for performance form scaling
PROMOTED_TRANSITION_W = 10    # Gameweek limit to transition promoted team standard model

# Data file paths
MOCK_INJURY_PATH = 'data/raw/injuries_mock.csv'
HISTORICAL_MATCHES_PATH = 'data/processed/matches_with_features.csv'
FIXTURES_PATH = 'data/raw/fixtures_raw.csv'
SQUAD_VALUES_PATH = 'data/raw/squad_values.csv'
MANAGERS_PATH = 'data/raw/managers.csv'
CALIBRATION_PARAMS_PATH = 'output/calibration_params.json'


def load_calibrated_temperature(fallback=CALIBRATION_TEMP, path=CALIBRATION_PARAMS_PATH):
    """Return latest dynamically fitted calibration temperature if persisted, else fallback."""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            temp = float(data.get('calibration_temp', fallback))
            if 0.5 <= temp <= 3.0:
                return temp
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return float(fallback)


# Set default v2 configuration mapping
def get_v2_config(use_saved_calibration=True):
    cal_temp = (
        load_calibrated_temperature()
        if (ENABLE_DYNAMIC_CALIBRATION and use_saved_calibration)
        else CALIBRATION_TEMP
    )
    return {
        'decay_phi': DECAY_PHI if ENABLE_TIME_DECAY else 0.0,
        'dc_weight': DC_WEIGHT,
        'dc_xg_weight': DC_XG_WEIGHT if ENABLE_DC_XG_BLEND else 0.0,
        'calibration_temp': cal_temp,
        'dynamic_calibration': ENABLE_DYNAMIC_CALIBRATION,
        'dynamic_elo': ENABLE_DYNAMIC_ELO,
        'promoted_handling': ENABLE_PROMOTED_HANDLING,
        'injuries': ENABLE_INJURIES,
        'congestion': ENABLE_CONGESTION,
        'bayesian_form': ENABLE_BAYESIAN_FORM,
        'form_boost': FORM_BOOST_RATE,
        'promoted_transition': PROMOTED_TRANSITION_W
    }
