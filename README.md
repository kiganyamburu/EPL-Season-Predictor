# EPL Season Predictor 2026/27 — Machine Learning & Simulation Engine

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-EB5424.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![SciPy](https://img.shields.io/badge/SciPy-1.10+-8CAAE6.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Pytest](https://img.shields.io/badge/Tests-32%20Passed-brightgreen.svg?logo=pytest&logoColor=white)](https://pytest.org/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF.svg?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade sports analytics and probabilistic forecasting system for the **2026/27 English Premier League (EPL)** season. Combining a time-discounted bivariate Dixon-Coles Poisson model blended with Expected Goals (xG), a gradient-boosted XGBoost classifier engineered with 13 match-level signals, dynamic out-of-sample temperature calibration, and a 10,000-iteration state-aware Monte Carlo simulation engine.

Delivered alongside an interactive 8-tab Streamlit dashboard, automated data ingestion pipelines (Fantasy Premier League API + `football-data.co.uk`), and scheduled CI/CD automation.

---

## Table of Contents

- [Key Highlights](#key-highlights)
- [System Architecture](#system-architecture)
- [Statistical & Machine Learning Models](#statistical--machine-learning-models)
  - [1. Time-Discounted Dixon-Coles Poisson Model with xG Blending](#1-time-discounted-dixon-coles-poisson-model-with-xg-blending)
  - [2. Multi-Class XGBoost Match Outcome Classifier](#2-multi-class-xgboost-match-outcome-classifier)
  - [3. Dynamic Out-of-Sample Temperature Calibration](#3-dynamic-out-of-sample-temperature-calibration)
  - [4. State-Aware Monte Carlo Season Simulation](#4-state-aware-monte-carlo-season-simulation)
- [Feature Engineering Pipeline](#feature-engineering-pipeline)
- [Live Data Feeds & Automation](#live-data-feeds--automation)
- [Streamlit Dashboard Walkthrough](#streamlit-dashboard-walkthrough)
- [Project Directory Structure](#project-directory-structure)
- [Quickstart & Usage](#quickstart--usage)
  - [Installation](#installation)
  - [Running the Streamlit Dashboard](#running-the-streamlit-dashboard)
  - [Running the Predictive Pipelines](#running-the-predictive-pipelines)
  - [Automated Weekly Refresh](#automated-weekly-refresh)
  - [Running the Test Suite](#running-the-test-suite)
- [CI/CD & Task Scheduling](#cicd--task-scheduling)
- [License](#license)

---

## Key Highlights

- **4,280 Historical Matches (12 Seasons)**: Trained on comprehensive match statistics from 2015/16 through the live 2026/27 campaign.
- **xG-Blended Parameter Estimation**: Blends continuous Expected Goals ($w_{\text{xG}} = 0.40$) into Dixon-Coles Poisson likelihood optimization to mitigate small-sample finishing variance while maintaining exact integer matrix low-score adjustments.
- **Multi-Spell Managerial Database**: Historical registry covering 234 verified managerial appointments across 35 clubs (2012–2026) to accurately model the "new manager bounce" across historical backtesting.
- **Transfer Window Impact Modeling**: Explicitly tracks squad market value percentiles, summer net spend, and incoming player impact ratings across all 20 clubs.
- **Live-State Post-GW10 Initialization**: Seamlessly incorporates real-world completed results, live points, and goal difference, running 10,000 Monte Carlo iterations from current standings with dynamic in-simulation Elo shifts and circular rolling form buffers.
- **Dynamic Walk-Forward Temperature Scaling**: Uses bounded Brent optimization (`scipy.optimize.minimize_scalar`) on out-of-sample validation data to eliminate overconfidence and minimize Brier score and multiclass log-loss ($T = 1.3952$).
- **Fully Automated Weekly Workflows**: Automated synchronization of live match results, fixture rescheduling, and squad injury lists via the official Fantasy Premier League (FPL) API and `football-data.co.uk`.

---

## System Architecture

```
                                  LIVE & HISTORICAL DATA FEEDS
                    ┌───────────────────────────┬───────────────────────────┐
                    │  football-data.co.uk API  │   Official FPL REST API   │
                    │  (Match stats, HS, SOT)   │ (Fixtures, Injuries, GWs) │
                    └─────────────┬─────────────┴─────────────┬─────────────┘
                                  │                           │
                                  ▼                           ▼
                     [data/raw/historical/*.csv]   [data/raw/injuries_mock.csv]
                     [data/raw/fixtures_raw.csv]   [data/raw/transfers.csv]
                                  │                           │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                    FEATURE ENGINEERING ENGINE
                                (src/feature_engineering.py)
                    ┌───────────────────────────────────────────────────────┐
                    │ • Dynamic Elo with MOV Multiplier & Seasonal Mean-Rev │
                    │ • 10-Match Rolling xG, xGA, SOT, & Red-Card Discipline│
                    │ • Squad Value Percentile & Net Spend Impact Ratings   │
                    │ • Multi-Spell Managerial Tenure & New-Manager Window  │
                    │ • Fixture Congestion & Midweek Rest Days              │
                    └───────────────────────────┬───────────────────────────┘
                                                │
                                                ▼
                                    HYBRID MODELING ENSEMBLE
                    ┌───────────────────────────┴───────────────────────────┐
                    ▼                                                       ▼
        DIXON-COLES POISSON (35%)                               XGBOOST CLASSIFIER (65%)
       (src/models.py: DixonColesModel)                        (src/models.py: XGBoostPredictor)
  • Time discounting (phi = 0.0012)                       • 13 match-level engineered signals
  • xG Blending (w_xg = 0.40)                             • Tree gradient boosting
  • Low-score dependence rho correction                   • Calibrated multiclass probabilities
                    │                                                       │
                    └───────────────────────────┬───────────────────────────┘
                                                │
                                                ▼
                                 DYNAMIC TEMPERATURE SCALING
                                (T = 1.3952 via Scipy minimize_scalar)
                                                │
                                                ▼
                                 STATE-AWARE MONTE CARLO SIMULATOR
                                    (src/pipeline_v2.py: 10,000 Runs)
                    ┌───────────────────────────────────────────────────────┐
                    │ • Seeded with completed fixtures (Points, GD, GF, GA) │
                    │ • Dynamic circular buffers for rolling team form      │
                    │ • In-season Elo adjustments per simulated match       │
                    └───────────────────────────┬───────────────────────────┘
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         ▼                                             ▼
               PREDICTION ARTIFACTS                           INTERACTIVE STREAMLIT APP
         • output/predictions.csv                            (app.py: 8 Dedicated Tabs)
         • output/final_table_projection.csv           • Live & Projected Tables
         • output/calibration_params.json              • Standing Distributions & Odds
         • output/evaluation_report.md                 • H2H Simulator & Scoreline Heatmaps
                                                       • What-If Counterfactual Sandbox
```

---

## Statistical & Machine Learning Models

### 1. Time-Discounted Dixon-Coles Poisson Model with xG Blending

Standard Poisson models assume independence between home and away goals and treat goals scored five years ago equally to those scored yesterday. The implemented Dixon-Coles framework directly resolves both limitations:

$$\lambda_k = \alpha_i \cdot \beta_j \cdot \gamma, \quad \mu_k = \alpha_j \cdot \beta_i$$

Where:
- $\alpha_i$ is the attacking strength of the home team.
- $\beta_j$ is the defensive vulnerability of the away team.
- $\gamma$ is the home field advantage parameter.
- $\lambda_k, \mu_k$ represent the expected goals for the home and away sides.

#### Low-Scoreline Correlation ($\rho$)
To adjust for the structural undercounting of low-scoring draws ($0\text{--}0$) and overcounting of $1\text{--}0, 0\text{--}1, 1\text{--}1$ scorelines, an adjustment factor $\tau_{\lambda, \mu}(x, y)$ is applied:

$$\tau_{\lambda, \mu}(x, y) = \begin{cases} 
1 - \lambda \mu \rho & \text{for } x = 0, y = 0 \\
1 + \lambda \rho & \text{for } x = 0, y = 1 \\
1 + \mu \rho & \text{for } x = 1, y = 0 \\
1 - \rho & \text{for } x = 1, y = 1 \\
1 & \text{otherwise}
\end{cases}$$

#### Time-Decay Weighting
Historical matches are discounted using an exponential weighting function:

$$w(t_k) = \exp(-\phi \cdot (t_{\text{current}} - t_k))$$

With $\phi = 0.0012$, corresponding to an empirical half-life of $\approx 577$ days (~1.5 seasons).

#### Continuous Expected Goals (xG) Blending
Rather than estimating attack and defense parameters strictly from integer goal tallies—which are vulnerable to finishing variance and goalkeeper hot-streaks—the objective function optimizes over blended goal expectations:

$$y_{i, \text{blend}} = (1 - w_{\text{xG}}) \cdot \text{FTHG}_i + w_{\text{xG}} \cdot \text{HxG}_i$$

Setting $w_{\text{xG}} = 0.40$ yields the lowest out-of-sample log-loss across validation seasons. Attack and defense parameters are identified under the constraint $\frac{1}{N} \sum_{i=1}^{N} \alpha_i = 1$ via L-BFGS-B optimization (`ftol=1e-7, gtol=1e-4`).

---

### 2. Multi-Class XGBoost Match Outcome Classifier

The XGBoost model operates on 13 feature dimensions per match to predict discrete outcome probabilities $(P_{\text{Home}}, P_{\text{Draw}}, P_{\text{Away}})$:

| Feature Name | Description | Source / Methodology |
|:---|:---|:---|
| `EloDiff` | Difference in pre-match Elo ratings | Dynamic Elo with MOV multiplier |
| `HomeSquadValPercentile` | Home team squad valuation rank (0.0 to 1.0) | Summer 2026 Transfermarkt values |
| `AwaySquadValPercentile` | Away team squad valuation rank (0.0 to 1.0) | Summer 2026 Transfermarkt values |
| `HomeTransferImpact` | Squad rating change from new signings | Market spend & incoming quality proxy |
| `AwayTransferImpact` | Squad rating change from new signings | Market spend & incoming quality proxy |
| `HomeRollingXG` | 10-match rolling Expected Goals created | Calibrated shot-to-xG formulation |
| `AwayRollingXG` | 10-match rolling Expected Goals created | Calibrated shot-to-xG formulation |
| `HomeRollingXGA` | 10-match rolling Expected Goals conceded | Calibrated shot-to-xG formulation |
| `AwayRollingXGA` | 10-match rolling Expected Goals conceded | Calibrated shot-to-xG formulation |
| `HomeRollingSOTFor` | 10-match rolling shots on target generated | `football-data.co.uk` HST feed |
| `AwayRollingSOTFor` | 10-match rolling shots on target generated | `football-data.co.uk` AST feed |
| `HomeRollingRedCards` | 10-match rolling red cards received | Pre-match disciplinary measure |
| `AwayRollingRedCards` | 10-match rolling red cards received | Pre-match disciplinary measure |

#### Historical Feature Importance (Top Predictors):
1. `EloDiff` (15.91%)
2. `HomeSquadValPercentile` (6.75%)
3. `HomeTransferImpact` (5.56%)
4. `HomeRollingXG` (4.94%)
5. `AwayRollingXG` (4.61%)

---

### 3. Dynamic Out-of-Sample Temperature Calibration

Standard tree-based ensembles and Poisson models often suffer from overconfidence in sports betting contexts. To guarantee calibrated probabilities, the raw log-odds $z_k = \log(P_k)$ are scaled by a global temperature parameter $T$:

$$P_k(T) = \frac{\exp(z_k / T)}{\sum_{m \in \{H, D, A\}} \exp(z_m / T)}$$

- The temperature parameter $T$ is automatically fitted on the prior completed validation season (`2025/26`, $n = 380$) using bounded 1D Brent optimization (`scipy.optimize.minimize_scalar(..., bounds=(0.5, 3.0), method='bounded')`).
- Empirical validation: On the 2025/26 season, dynamic temperature calibration selected $T = 1.3952$, reducing multiclass log-loss from **1.0421** to **1.0325** and lowering Brier score from **0.5901** to **0.5847**.
- Calibrated values are persisted to `output/calibration_params.json` and loaded during runtime.

---

### 4. State-Aware Monte Carlo Season Simulation

The Monte Carlo simulation engine (`src/pipeline_v2.py`) simulates the remainder of the 2026/27 campaign across **10,000 independent stochastic runs**:

1. **State Initialization**: Automatically loads completed fixtures from `data/raw/fixtures_raw.csv`. Each simulation begins with the true current league table (actual points, goal differences, goals scored, and records).
2. **In-Season Dynamic Elo Adjustments**: When a simulated fixture finishes, both teams' Elo ratings shift dynamically based on the simulated margin of victory:
   $$K_{\text{adj}} = K \cdot (1 + 0.5 \cdot \max(0, |\Delta \text{Goals}| - 1))$$
3. **Circular Rolling Form Updates**: Teams maintain 10-match circular buffers for rolling goals, xG, and shots on target that update after every simulated gameweek.
4. **Tiebreakers**: Applies official Premier League ranking rules: Points $\rightarrow$ Goal Difference $\rightarrow$ Goals Scored $\rightarrow$ Head-to-Head Record.
5. **Output Aggregation**: Tracks finishing position frequencies, expected points ($\mathbb{E}[\text{Pts}]$), championship odds, UEFA Champions League / Europa League qualification odds, and relegation probabilities.

---

## Feature Engineering Pipeline

### Dynamic Elo Rating System
- Initialized from historical match results dating back to 2015.
- Inter-season regression to the mean:
  $$R_{\text{new}} = (1 - c) \cdot R_{\text{old}} + c \cdot \mu_{\text{league}}, \quad c = 0.20$$
- Promoted teams enter with a calibrated baseline Elo ($R_{\text{promoted}} = 1420$) blended with early-season results over their opening 10 matches.

### Unbiased Historical Shots-to-xG Formula
Prior to the wide availability of tracking-data xG in older seasons (2015–2020), an empirically fitted formulation is applied:

$$\text{xG}_{\text{est}} = 0.25 \cdot \text{HST} + 0.04 \cdot \max(0, \text{HS} - \text{HST})$$

Validation against modern tracking data confirms an average of **1.568 HxG** estimated vs. **1.566 HxG** actual ($r = 0.692$).

### Multi-Spell Managerial Tenure Database
- Located at `data/raw/historical_managers.csv` covering **234 managerial spells** across 35 clubs.
- Tracks multi-spell managers (e.g., David Moyes, Roy Hodgson, Eddie Howe) using bisection time search.
- Computes `NewManager` binary flag (active for 180 days post-appointment) and tenure in days.

---

## Live Data Feeds & Automation

The system integrates real-time automated data collection via `scripts/weekly_refresh.py`:

```bash
# Sync live FPL fixtures, results, and injuries, then update forecasts
python scripts/weekly_refresh.py
```

1. **FPL REST API Ingestion**:
   - `https://fantasy.premierleague.com/api/fixtures/`: Retrieves kickoff times, completed match scores, and gameweek statuses.
   - `https://fantasy.premierleague.com/api/bootstrap-static/`: Extracts live player availability, injuries, suspensions, and return dates directly into `data/raw/injuries_mock.csv`.
2. **`football-data.co.uk` Ingestion**:
   - Downloads `E0.csv` for detailed matchday statistics (shots, shots on target, corners, fouls, and red cards).
3. **Automated Artifact Snapshots**:
   - Stores point trajectories and rank histories in `output/standings_history.csv` and `output/predictions_history.csv`.

---

## Streamlit Dashboard Walkthrough

Launch the web app locally with:

```bash
streamlit run app.py
```

The application provides an 8-tab analytical workspace:

| Tab | Name | Capabilities & Features |
|:---:|:---|:---|
| **1** | **Standings (Live & Projected)** | Toggle between Projected Table, Actual Live Standings, and Live vs. Projected Delta. Glassmorphic KPI cards for Title, Top 4, and Relegation lines. |
| **2** | **Standing Distributions** | Interactive violin and box plots showing point distributions and 1st–20th position probabilities across 10,000 simulations. |
| **3** | **Match Predictions** | Fixture explorer for all 380 matches. Filter by Gameweek or Club. Displays calibrated 1X2 probabilities, fair decimal odds, and contextual factor tooltips. |
| **4** | **Head-to-Head Simulator** | Custom matchup analyzer. Generates a 6x6 bivariate Poisson scoreline probability heatmap and win/draw/loss distributions for any two clubs. |
| **5** | **Squad Transfers & Signings** | Summer 2026/27 transfer expenditure analysis, net spend rankings, and squad market valuation percentiles. |
| **6** | **Narrative Insights** | Natural language matchday analysis, highlighting title race pressure, over/under-performing teams, and relegation scrap dynamics. |
| **7** | **Model Calibration & Accuracy** | Empirical reliability curves, Brier score tracking, log-loss monitoring, and out-of-sample temperature calibration verification. |
| **8** | **What-If Sandbox** | Interactive counterfactual sandbox. Adjust managerial appointments ("new manager bounce"), injury crises, European fatigue, and Dixon-Coles vs XGBoost blend weights with real-time re-simulation. |

---

## Project Directory Structure

```text
EPL-Season-Predictor/
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Continuous integration test runner (Pytest on 3.12 & 3.14)
│       └── weekly_refresh.yml         # Scheduled Tuesday cron for data sync and simulation
├── assets/
│   └── dashboard_preview.png          # Dashboard preview illustration
├── data/
│   ├── processed/
│   │   ├── final_elos.csv             # Computed pre-match and current Elo ratings
│   │   ├── historical_results.csv     # Cleaned 2015–2026 match results database
│   │   ├── matches_with_features.csv  # 4,280 matches with 13 engineered features
│   │   └── unplayed_fixtures.csv      # Remaining 2026/27 fixtures schedule
│   └── raw/
│       ├── historical/                # Historical E0.csv files (2015/16 - 2025/26)
│       ├── fixtures_raw.csv           # 2026/27 fixture schedule and completed results
│       ├── historical_managers.csv    # 234 historical manager spells (2012–2026)
│       ├── injuries_mock.csv          # Live FPL injury & squad availability feed
│       ├── managers.csv               # Current active 2026/27 manager records
│       ├── premier-league.ics         # Official fixture calendar feed
│       ├── squad_values.csv           # Squad market valuations
│       └── transfers.csv              # Summer 2026 transfer window net spend & impact
├── output/
│   ├── calibration_params.json        # Out-of-sample dynamic temperature calibration (T)
│   ├── evaluation_report.csv          # Backtesting evaluation metrics table
│   ├── evaluation_report.md           # Comparative backtesting markdown report
│   ├── final_table_projection.csv     # Monte Carlo 10k final simulated standings & odds
│   ├── predictions.csv                # Calibrated 1X2 match probabilities for all 380 games
│   ├── predictions_history.csv        # Historical match predictions archive
│   └── standings_history.csv          # Weekly league table snapshot history
├── scripts/
│   ├── register_weekly_task.ps1       # Windows Task Scheduler registration script
│   ├── run_weekly_job.ps1             # Local automation runner script
│   └── weekly_refresh.py              # Live FPL data sync, pipeline execution, and snapshot tool
├── src/
│   ├── v2/
│   │   ├── bayesian_form.py           # Autoregressive form-boosting and mean-reversion
│   │   ├── calibration.py             # Temperature scaling and Brier score evaluation
│   │   ├── config.py                  # Global hyperparameters, paths, and model toggles
│   │   ├── congestion.py              # Rest day calculations and European fixture fatigue
│   │   ├── elo_dynamic.py             # In-season dynamic Elo mathematics
│   │   ├── explain.py                 # Natural language narrative generation
│   │   ├── injuries.py                # Squad availability and xG injury scaling
│   │   ├── market.py                  # Betting market benchmark parser & de-vigging
│   │   └── promoted.py                # Promoted team transition blending
│   ├── data_collection.py             # Data scrapers, FPL API client, and schedule parsers
│   ├── feature_engineering.py         # Dynamic Elo, rolling stats, SOT, cards, and tenures
│   ├── models.py                      # Dixon-Coles Poisson & XGBoost Predictor classes
│   ├── pipeline.py                    # Legacy v1 baseline pipeline
│   ├── pipeline_v2.py                 # Orchestration pipeline for v2 training & simulation
│   └── simulation.py                  # State-aware 10,000-run Monte Carlo simulation engine
├── tests/
│   ├── test_data_integrity.py         # Validates files, schema types, ranges, and consistency
│   └── test_features_and_models.py    # Unit tests for Dixon-Coles, XGBoost, and temperature scaling
├── app.py                             # Interactive 8-tab Streamlit dashboard
├── requirements.txt                   # Production Python package dependencies
└── README.md                          # Comprehensive system documentation
```

---

## Quickstart & Usage

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/EPL-Season-Predictor.git
   cd EPL-Season-Predictor
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

### Running the Streamlit Dashboard

Launch the dashboard locally:

```bash
streamlit run app.py
```

The web application will open in your browser at `http://localhost:8501`.

---

### Running the Predictive Pipelines

Execute the full end-to-end model training, validation sweep, and 10,000 Monte Carlo simulations:

```bash
# Run complete v2 pipeline (including feature engineering & backtest sweeps)
python -m src.pipeline_v2

# Fast run (skip cross-validation sweeps, execute training & 10k simulations)
python -m src.pipeline_v2 --skip-sweeps
```

To run individual pipeline stages separately:

```bash
# Stage 1: Data Collection & Historical Sync
python src/data_collection.py

# Stage 2: Feature Engineering (Elo, Rolling xG/SOT/Cards, Manager Tenures)
python -m src.feature_engineering
```

---

### Automated Weekly Refresh

To simulate a gameweek transition or synchronize live completed matches and injuries from the official FPL API:

```bash
# Fetch live data from FPL API and run pipeline
python scripts/weekly_refresh.py

# Simulate completing Gameweek 12 with synthetic scores (if offline)
python scripts/weekly_refresh.py --gameweek 12
```

---

### Running the Test Suite

The repository includes a comprehensive 32-test unit and regression suite:

```bash
# Run all tests
pytest -v

# Run data integrity tests only
pytest tests/test_data_integrity.py -v

# Run model and feature engineering tests only
pytest tests/test_features_and_models.py -v
```

---

## CI/CD & Task Scheduling

### GitHub Actions
The repository includes automated CI/CD workflows under `.github/workflows/`:
- **`ci.yml`**: Runs on every `push` and `pull_request` to `main`, validating the full test suite across Python 3.12 and 3.14.
- **`weekly_refresh.yml`**: Runs on a scheduled cron every Tuesday at 06:00 UTC to automatically sync live results, re-calibrate models, and update projected standings.

### Windows Task Scheduler
For local Windows servers or workstations, automated scheduling scripts are provided in `scripts/`:
- **`scripts/run_weekly_job.ps1`**: PowerShell script executing `weekly_refresh.py` and logging outputs to `output/weekly_refresh.log`.
- **`scripts/register_weekly_task.ps1`**: Automatically registers a weekly scheduled task in Windows Task Scheduler:
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/register_weekly_task.ps1
  ```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
