# EPL Season Predictor 2026/27 — Upgraded v2

A machine learning system that predicts outcomes for every match in the upcoming 2026/27 English Premier League (EPL) season, simulates the full season using Monte Carlo methods, and outputs match-by-match probabilities, final projected tables, and interactive sandboxes.

## Project Structure
```text
EPL-Season-Predictor/
├── data/
│   ├── raw/                  # Raw downloaded files, scraped fixtures, and squad values
│   │   ├── injuries_mock.csv # Mock injuries list for 2026/27 season
│   │   └── managers.csv      # Manager appointment dates
│   └── processed/            # Engineered features and training datasets
├── output/
│   ├── predictions.csv       # Matchday probabilities for all 380 fixtures (with explanations)
│   ├── final_table_projection.csv # Simulated final standings & odds
│   ├── predictions_history.csv  # Saved predictions snapshot over gameweeks
│   └── evaluation_report.md  # Backtesting metrics comparing v2 configs
├── src/
│   ├── data_collection.py    # Fetch and parse historical data, squad values, and fixtures
│   ├── feature_engineering.py # Calculate Elo, Attack/Defense strengths, manager variance, etc.
│   ├── models.py             # Dixon-Coles model class and XGBoost classifier setup
│   ├── pipeline_v2.py        # Orchestrates the complete end-to-end v2 flow
│   └── v2/                   # Modular feature upgrades
│       ├── config.py         # Center for configurations and toggles
│       ├── injuries.py       # Player injury tracking and Dixon-Coles expected goals modifiers
│       ├── congestion.py     # Midweek European & Domestic Cup congestion fatigue calculations
│       ├── promoted.py       # Linear transition blending from promoted baseline parameters
│       ├── elo_dynamic.py    # Dynamic in-season Elo update mathematics
│       ├── market.py         # Betting market benchmark parser and de-vig metric scoring
│       ├── bayesian_form.py  # Autoregressive form-boosting and mean-reversion simulation
│       └── explain.py        # Natural language explainability generator
├── scripts/
│   └── weekly_refresh.py     # Gameweek progress emulator and snapshot snapshotting tool
├── app.py                    # Streamlit interactive dashboard (with What-If Sandbox)
├── README.md                 # Project explanation and run instructions
└── requirements.txt          # Python dependencies
```

## How to Run the Pipeline

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Collection
This downloads the historical E0.csv files (2015/16 to 2025/26) from `football-data.co.uk`, fetches the 2026/27 schedule from the `.ics` calendar feed, parses it, and sets up metadata for squads and managers.
```bash
python src/data_collection.py
```

### 3. Run Feature Engineering
This computes rolling Elo ratings (with season-to-season regression and promoted team baselines), rolling goals and form, and merges squad market value ranks and manager changes.
```bash
python src/feature_engineering.py
```

### 4. Run Backtesting sweeps and Simulation Projections
This runs the backtesting suite comparing all 8 configuration changes against the 2024/25 season, outputs the comparative metrics to `output/evaluation_report.md`, trains final models, and simulates 10,000 runs of the 2026/27 season using stateful Monte Carlo modeling.
```bash
python -m src.pipeline_v2
```

### 5. Launch the Dashboard
Run the Streamlit application to view interactive standings, points and rank probability ranges, match prediction odds with factor breakdown tooltips, and the What-If sandbox playground:
```bash
streamlit run app.py
```

### 6. Run Gameweek Weekly Refresh
Simulate a completed gameweek (e.g. Gameweek 1) using the automated emulator to move fixtures to completed matches with mock scores, capture prediction snapshots, and refresh all projected standings:
```bash
python scripts/weekly_refresh.py --gameweek 1
```

---

## Technical Details

### 1. Dixon-Coles Poisson Model
- Models expected goals as Poisson distributions adjusted for low scorelines (0-0, 1-0, 0-1, 1-1) using a correlation parameter $\rho$.
- Attacking ($\alpha$) and defensive ($\beta$) parameters are learned for all teams via maximum likelihood.
- Employs time discounting ($\exp(-\phi \cdot t)$) to discount historical matches and prioritize recent form.

### 2. XGBoost Match Outcome Classifier
- A gradient-boosted tree classifier trained on match-level features: Elo difference, rolling goals scored/conceded, rolling form points, squad value rank, and manager change flags.
- Outputs probabilistic distributions for Home Win, Draw, and Away Win.

### 3. v2 Predictive Feature Upgrades
- **Dynamic Elo Updates**: Tracks rating shifts in-season using goal difference multipliers (analogous to World Football Elo).
- **Injury Adjustments**: Computes squad availability percentage dynamically. Attacking expectations are scaled down and defensive vulnerability scales up in proportion to missing player ratings.
- **Fatigue Congestion**: Flags fixtures scheduled with 3 or fewer days of rest (and overlays European midweek/Carabao Cup schedules for cup-congested top clubs).
- **Promoted Blending**: Blends promoted teams' expectations towards historical averages over their first 10 gameweeks.
- **Bayesian Form**: Models streak/slump form multipliers that revert slowly to the mean inside each simulation run.
- **Explainability**: Generates concise, bulleted textual reasoning summarizing ELO differences, rest fatigue, and key injuries.
