import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# ==============================================================================
# 1. DIXON-COLES POISSON MODEL IMPLEMENTATION
# ==============================================================================

class DixonColesModel:
    def __init__(self, time_decay_phi=0.003, xg_weight=0.4):
        self.phi = time_decay_phi
        self.xg_weight = xg_weight
        self.teams = []
        self.team_indices = {}
        self.attack_params = None
        self.defense_params = None
        self.home_advantage = 1.0
        self.rho = 0.0
        
    def _dixon_coles_adjustment(self, x, y, lmbda, mu, rho):
        """Adjustment factor for low scorelines (0-0, 1-0, 0-1, 1-1)."""
        if x == 0 and y == 0:
            return 1.0 - lmbda * mu * rho
        elif x == 1 and y == 0:
            return 1.0 + mu * rho
        elif x == 0 and y == 1:
            return 1.0 + lmbda * rho
        elif x == 1 and y == 1:
            return 1.0 - rho
        else:
            return 1.0

    @staticmethod
    def _extract_effective_xg(df, side):
        """Return actual HxG/AxG when present, else calibrated shots-xG, else raw goals."""
        xg_col = 'HxG' if side == 'H' else 'AxG'
        s_col, st_col, g_col = ('HS', 'HST', 'FTHG') if side == 'H' else ('AS', 'AST', 'FTAG')
        out = df[g_col].astype(float).copy()
        if st_col in df.columns and s_col in df.columns:
            has_shots = df[st_col].notna() & df[s_col].notna()
            sot = df.loc[has_shots, st_col].astype(float)
            off_target = np.maximum(0.0, df.loc[has_shots, s_col].astype(float) - sot)
            out.loc[has_shots] = 0.25 * sot + 0.04 * off_target
        if xg_col in df.columns:
            has_xg = df[xg_col].notna()
            out.loc[has_xg] = df.loc[has_xg, xg_col].astype(float)
        return np.clip(out.to_numpy(dtype=float), 0.0, None)

    def fit(self, df, xg_weight=None):
        """
        Fit Dixon-Coles parameters: attack, defense, home advantage, and rho.
        Blends actual goals (FTHG/FTAG) with HxG/AxG (or shots-based xG) for rate
        estimation while preserving integer scorelines for low-score rho correlation.
        """
        # Get unique list of teams
        self.teams = sorted(list(set(df['HomeTeam']).union(set(df['AwayTeam']))))
        self.team_indices = {team: i for i, team in enumerate(self.teams)}
        num_teams = len(self.teams)
        
        # Calculate time weights: exp(-phi * days_elapsed)
        # We calculate days elapsed relative to the most recent match date in df
        parsed_dates = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
        max_date = parsed_dates.max()
        df = df.copy()
        df['DaysElapsed'] = (max_date - parsed_dates).dt.days
        df['Weight'] = np.exp(-self.phi * df['DaysElapsed'])
        
        # Format match data for rapid optimization
        home_idx = df['HomeTeam'].map(self.team_indices).values
        away_idx = df['AwayTeam'].map(self.team_indices).values
        home_goals = df['FTHG'].values.astype(int)
        away_goals = df['FTAG'].values.astype(int)

        w_xg = self.xg_weight if xg_weight is None else float(xg_weight)
        h_xg = self._extract_effective_xg(df, 'H')
        a_xg = self._extract_effective_xg(df, 'A')
        target_home = np.clip((1.0 - w_xg) * home_goals + w_xg * h_xg, 0.0, None)
        target_away = np.clip((1.0 - w_xg) * away_goals + w_xg * a_xg, 0.0, None)
        weights = df['Weight'].values
        
        # Precompute log-gamma for continuous/integer Poisson log-probability calculations
        from scipy.special import gammaln
        log_fact_home = gammaln(target_home + 1)
        log_fact_away = gammaln(target_away + 1)
        
        # Initial parameters: 
        # attack parameters (first num_teams params, default 1.0)
        # defense parameters (next num_teams params, default -1.0)
        # home advantage (1 param, default 0.25 in log-space)
        # rho (1 param, default 0.0)
        init_params = np.concatenate([
            np.ones(num_teams),       # Attack (all optimized freely)
            -np.ones(num_teams),      # Defense (log-defense, so negative is better defense)
            [0.25],                   # Log-Home Advantage
            [0.0]                     # Rho
        ])
        
        # Objective: Negative Log-Likelihood
        def negative_log_likelihood(params):
            alpha = params[:num_teams]
            beta = np.exp(params[num_teams:2*num_teams]) # exponentiate to keep positive
            gamma = np.exp(params[2*num_teams])          # exponentiate to keep positive
            rho = params[-1]
            
            # Constraint penalty: Mean of alpha should be 1.0
            penalty = 10000.0 * (np.mean(alpha) - 1.0)**2
            
            # Dixon-Coles parameters for each match
            lmbda = alpha[home_idx] * beta[away_idx] * gamma
            mu = alpha[away_idx] * beta[home_idx]
            
            # Catch zero values to prevent log(0)
            lmbda = np.clip(lmbda, 1e-6, None)
            mu = np.clip(mu, 1e-6, None)
            
            # Compute adjustment factor for integer low scoring matches (vectorized)
            adj = np.ones_like(home_goals, dtype=float)
            
            # Mask for (0, 0)
            mask_00 = (home_goals == 0) & (away_goals == 0)
            adj[mask_00] = 1.0 - lmbda[mask_00] * mu[mask_00] * rho
            
            # Mask for (1, 0)
            mask_10 = (home_goals == 1) & (away_goals == 0)
            adj[mask_10] = 1.0 + mu[mask_10] * rho
            
            # Mask for (0, 1)
            mask_01 = (home_goals == 0) & (away_goals == 1)
            adj[mask_01] = 1.0 + lmbda[mask_01] * rho
            
            # Mask for (1, 1)
            mask_11 = (home_goals == 1) & (away_goals == 1)
            adj[mask_11] = 1.0 - rho
            
            # Clip adjustment factor to prevent flat gradients or negative values
            adj = np.clip(adj, 0.01, 10.0)
            
            # Compute Poisson log-likelihoods using goals/xG blended targets
            log_p_home = -lmbda + target_home * np.log(lmbda) - log_fact_home
            log_p_away = -mu + target_away * np.log(mu) - log_fact_away
            log_adj = np.log(adj)
            
            log_probs = log_p_home + log_p_away + log_adj
            
            # Weighted negative log-likelihood
            nll = -np.sum(weights * log_probs) + penalty
            return nll
            
        # Optimization bounds
        bounds = []
        # attack bounds
        bounds.extend([(0.1, 10.0) for _ in range(num_teams)])
        # defense bounds (log-space: -5 to 2)
        bounds.extend([(-5.0, 2.0) for _ in range(num_teams)])
        # log home advantage bounds
        bounds.append((-2.0, 2.0))
        # rho bounds
        bounds.append((-0.3, 0.3))
        
        # Perform optimization with tight ftol so small and large datasets alike converge beyond the mean-penalty direction
        print("Optimizing Dixon-Coles parameters...")
        res = minimize(
            negative_log_likelihood,
            init_params,
            bounds=bounds,
            method='L-BFGS-B',
            options={'ftol': 1e-7, 'gtol': 1e-4},
        )
        
        if res.success:
            print("Optimization successful!")
            # Extract parameters
            params = res.x
            self.attack_params = params[:num_teams]
            
            # Normalize attacks so they average to 1.0 exactly
            mean_attack = np.mean(self.attack_params)
            self.attack_params = self.attack_params / mean_attack
            
            # Adjust defense parameters for attack scaling
            self.defense_params = np.exp(params[num_teams:2*num_teams]) * mean_attack
            self.home_advantage = np.exp(params[2*num_teams])
            self.rho = params[-1]
            
            # Print average stats
            print(f"Global Home Advantage: {self.home_advantage:.3f}")
            print(f"Low Score Correlation (Rho): {self.rho:.4f}")
        else:
            print(f"Optimization FAILED: {res.message}. Falling back to default values.")
            self.attack_params = np.ones(num_teams)
            self.defense_params = np.ones(num_teams)
            self.home_advantage = 1.25
            self.rho = 0.0
            
    def predict_match_probabilities(self, home_team, away_team, max_goals=10):
        """
        Predict Home Win, Draw, and Away Win probabilities for a match.
        Also returns the most likely scoreline.
        """
        # Fallbacks if teams are unrecognized
        h_idx = self.team_indices.get(home_team)
        a_idx = self.team_indices.get(away_team)
        
        # Baseline rates
        if h_idx is not None and a_idx is not None:
            lmbda = self.attack_params[h_idx] * self.defense_params[a_idx] * self.home_advantage
            mu = self.attack_params[a_idx] * self.defense_params[h_idx]
        else:
            # Fallback to general average
            lmbda = 1.35
            mu = 1.15
            
        lmbda = max(lmbda, 1e-6)
        mu = max(mu, 1e-6)
        
        # Construct exact score probability matrix (up to max_goals x max_goals)
        score_matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        p_home_poisson = poisson.pmf(np.arange(max_goals + 1), lmbda)
        p_away_poisson = poisson.pmf(np.arange(max_goals + 1), mu)
        
        for x in range(max_goals + 1):
            for y in range(max_goals + 1):
                adj = self._dixon_coles_adjustment(x, y, lmbda, mu, self.rho)
                score_matrix[x, y] = p_home_poisson[x] * p_away_poisson[y] * adj
                
        # Normalize matrix so probabilities sum to 1
        score_matrix /= score_matrix.sum()
        
        # Aggregate outcomes (rows x = home goals, cols y = away goals)
        p_home_win = np.sum(np.tril(score_matrix, -1))  # x > y
        p_draw = np.sum(np.diag(score_matrix))          # x == y
        p_away_win = np.sum(np.triu(score_matrix, 1))   # x < y
        
        # Find most likely score
        max_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        most_likely_score = f"{max_idx[0]}-{max_idx[1]}"
        
        return {
            'HomeWin': p_home_win,
            'Draw': p_draw,
            'AwayWin': p_away_win,
            'ExpectedHomeGoals': lmbda,
            'ExpectedAwayGoals': mu,
            'MostLikelyScore': most_likely_score,
            'ScoreMatrix': score_matrix
        }

# ==============================================================================
# 2. XGBOOST MATCH CLASSIFIER IMPLEMENTATION
# ==============================================================================

class XGBoostPredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        # Features used for classification
        self.features = [
            'EloDiff', 
            'HomeRollingGoalsScored', 'HomeRollingGoalsConceded', 'HomeRollingPoints',
            'AwayRollingGoalsScored', 'AwayRollingGoalsConceded', 'AwayRollingPoints',
            'HomeSquadValPercentile', 'AwaySquadValPercentile',
            'HomeTransferImpact', 'AwayTransferImpact',
            'HomeNetSpend', 'AwayNetSpend',
            'HomeNewManager', 'AwayNewManager',
            'HomePromoted', 'AwayPromoted'
        ]

    def fit(self, train_df):
        """Train the XGBoost match outcome classifier."""
        X = train_df[self.features]
        # Target: FTR (Home, Draw, Away)
        y = self.label_encoder.fit_transform(train_df['FTR'])
        # Map: 0 = 'A' (Away Win), 1 = 'D' (Draw), 2 = 'H' (Home Win)
        # Let's check classes order
        # label_encoder.classes_ should be ['A', 'D', 'H']
        
        print("Training XGBoost Classifier...")
        self.model = xgb.XGBClassifier(
            max_depth=2,
            learning_rate=0.03,
            n_estimators=200,
            min_child_weight=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        )
        self.model.fit(X, y)
        print("XGBoost training completed.")
        
    def predict_probabilities(self, X_eval):
        """Predict match probabilities using the trained model."""
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        probs = self.model.predict_proba(X_eval[self.features])
        # Map output to dict matching Dixon-Coles structure
        # Column 0 corresponds to class 0 ('A'), Column 1 to class 1 ('D'), Column 2 to class 2 ('H')
        return {
            'AwayWin': probs[:, 0],
            'Draw': probs[:, 1],
            'HomeWin': probs[:, 2]
        }

# ==============================================================================
# 3. ENSEMBLE BLENDED SYSTEM
# ==============================================================================

def blend_predictions(dixon_coles_probs, xgboost_probs, dc_weight=0.5):
    """
    Linearly blend Dixon-Coles and XGBoost probabilities.
    dixon_coles_probs: dict containing 'HomeWin', 'Draw', 'AwayWin'
    xgboost_probs: dict containing 'HomeWin', 'Draw', 'AwayWin'
    """
    p_h = dc_weight * dixon_coles_probs['HomeWin'] + (1.0 - dc_weight) * xgboost_probs['HomeWin']
    p_d = dc_weight * dixon_coles_probs['Draw'] + (1.0 - dc_weight) * xgboost_probs['Draw']
    p_a = dc_weight * dixon_coles_probs['AwayWin'] + (1.0 - dc_weight) * xgboost_probs['AwayWin']
    
    # Re-normalize to sum to 1 exactly
    total = p_h + p_d + p_a
    return {
        'HomeWin': p_h / total,
        'Draw': p_d / total,
        'AwayWin': p_a / total
    }

def apply_temperature_scaling(probs, temperature=1.0):
    """
    Apply temperature scaling to a probability array of shape (..., 3)
    to calibrate probability sharpness without altering argmax rankings.
    """
    if temperature is None or abs(temperature - 1.0) < 1e-6:
        return probs
    clipped = np.clip(probs, 1e-8, 1.0)
    logits = np.log(clipped) / temperature
    logits -= np.max(logits, axis=-1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / np.sum(scaled, axis=-1, keepdims=True)

def fit_temperature_scaler(probs, true_labels, bounds=(0.5, 3.0)):
    """Fit scalar temperature T minimizing log-loss on validation probabilities."""
    from scipy.optimize import minimize_scalar

    probs_arr = np.asarray(probs, dtype=float)
    labels_arr = np.asarray(true_labels, dtype=int)
    if probs_arr.ndim != 2 or len(labels_arr) == 0 or probs_arr.shape[0] != len(labels_arr):
        return 1.0

    # Precompute log-probabilities directly so T=1.0 short-circuiting cannot flatten gradients
    logits = np.log(np.clip(probs_arr, 1e-8, 1.0))
    row_idx = np.arange(len(labels_arr))

    def _nll(temp):
        scaled_logits = logits / float(temp)
        scaled_logits -= np.max(scaled_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(scaled_logits)
        scaled = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        p_true = np.clip(scaled[row_idx, labels_arr], 1e-12, 1.0)
        return float(-np.mean(np.log(p_true)))

    res = minimize_scalar(_nll, bounds=bounds, method='bounded')
    if not res.success or _nll(res.x) >= _nll(1.0) - 1e-9:
        return 1.0
    return float(res.x)
