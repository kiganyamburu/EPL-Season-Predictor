# EPL Season Predictor 2026/27

A machine learning system that predicts outcomes for every match in the upcoming 2026/27 English Premier League (EPL) season, simulates the full season using Monte Carlo methods, and outputs win/draw/loss probabilities and league tables.

## Project Structure
```text
EPL-Season-Predictor/
├── data/
│   ├── raw/                  # Raw downloaded files, scraped fixtures, and squad values
│   └── processed/            # Engineered features and training datasets
├── output/
│   ├── predictions.csv       # Matchday probabilities for all 380 fixtures
│   └── final_table_projection.csv # Simulated final standings & odds
├── src/
│   ├── data_collection.py    # Fetch and parse historical data, squad values, and fixtures
│   ├── feature_engineering.py # Calculate Elo, Attack/Defense strengths, manager variance, etc.
│   ├── models.py             # Dixon-Coles model class and XGBoost classifier setup
│   ├── simulation.py         # Monte Carlo simulation loop
│   └── pipeline.py           # Orchestrates the complete end-to-end flow
├── app.py                    # Streamlit interactive dashboard
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

### 4. Run Backtesting and Simulation
This runs the backtesting suite on the 2024/25 season (evaluating Log-Loss and Accuracy), trains the final models on the complete historical dataset, and simulates the 2026/27 season 10,000 times using a highly optimized vectorized discrete sampler.
```bash
python -m src.pipeline
```

### 5. Launch the Dashboard
Run the Streamlit application to view interactive stands, points and rank probability ranges, and match prediction odds:
```bash
streamlit run app.py
```

---

## Technical Details

### 1. Dixon-Coles Poisson Model
- Models expected goals as Poisson distributions adjusted for low scorelines (0-0, 1-0, 0-1, 1-1) using a correlation parameter $\rho$.
- Attacking ($\alpha$) and defensive ($\beta$) parameters are learned for all teams via maximum likelihood.
- Employs time discounting ($\exp(-\phi \cdot t)$) to discount historical matches and prioritize recent form.

### 2. XGBoost Match Outcome Classifier
- A gradient boosted tree classifier trained on match-level features: Elo difference, rolling goals scored/conceded, rolling form points, squad value rank, and manager change flags.
- Outputs probabilistic distributions for Home Win, Draw, and Away Win.

### 3. Blended Ensemble
- Linearly blends the outputs of the Dixon-Coles and XGBoost models (50/50) to predict match outcome probabilities.
- Quadrants of the Dixon-Coles score matrix are rescaled to match the blended outcome probabilities, preserving the score-level Poisson distribution.

### 4. Optimized Season Simulation
- Simulates all 380 matches in the season.
- Employs a vectorized random sampler using pre-computed cumulative sum matrices, completing 10,000 full-season simulations in under a second.
- Tabulates points (3 for win, 1 for draw, 0 for loss), goal difference, and goals scored, ranking teams to compute title, top 4, and relegation odds.
