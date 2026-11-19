import os
import re
import pandas as pd
import pytest

from src.data_collection import clean_team_name, sync_live_injuries, TEAM_NAME_MAP


@pytest.mark.parametrize(
    "raw_name,expected",
    [
        ("Arsenal FC", "Arsenal"),
        ("AFC Bournemouth", "Bournemouth"),
        ("Brighton & Hove Albion FC", "Brighton"),
        ("Coventry City", "Coventry"),
        ("Hull City AFC", "Hull"),
        ("Ipswich Town", "Ipswich"),
        ("Leeds United", "Leeds"),
        ("Man Utd", "Man United"),
        ("Manchester United FC", "Man United"),
        ("Nottingham Forest", "Nott'm Forest"),
        ("Nottm Forest", "Nott'm Forest"),
        ("Spurs", "Tottenham"),
        ("Tottenham Hotspur", "Tottenham"),
    ],
)
def test_team_name_normalization(raw_name, expected):
    assert clean_team_name(raw_name) == expected
    assert clean_team_name(f"  {raw_name}  ") == expected


def test_team_name_map_values_are_idempotent():
    for canonical in set(TEAM_NAME_MAP.values()):
        assert clean_team_name(canonical) == canonical


def test_fixtures_raw_schema_and_schedule_integrity():
    path = "data/raw/fixtures_raw.csv"
    assert os.path.exists(path), f"Missing {path}"
    df = pd.read_csv(path)

    required_cols = {"MatchDay", "Date", "HomeTeam", "AwayTeam", "Summary"}
    assert required_cols.issubset(df.columns)
    assert len(df) == 380

    # Parseable dates and valid MatchDay integers 1..38
    pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="raise")
    assert df["MatchDay"].notna().all()
    assert df["MatchDay"].between(1, 38).all()

    # 20 unique clubs, each plays 19 home and 19 away matches with no self-matches
    home_counts = df["HomeTeam"].value_counts()
    away_counts = df["AwayTeam"].value_counts()
    assert len(home_counts) == 20
    assert len(away_counts) == 20
    assert (home_counts == 19).all()
    assert (away_counts == 19).all()
    assert (df["HomeTeam"] != df["AwayTeam"]).all()

    # Canonical names used in fixtures
    for team in set(df["HomeTeam"]).union(set(df["AwayTeam"])):
        assert clean_team_name(team) == team

    # Any scored summary has valid non-negative integer goals
    scored = df["Summary"].astype(str).str.extract(r"\((\d+)-(\d+)\)")
    has_score = scored.notna().all(axis=1)
    assert has_score.sum() >= 100


def test_reference_metadata_files_canonical_and_valid():
    squads = pd.read_csv("data/raw/squad_values.csv")
    managers = pd.read_csv("data/raw/managers.csv")
    injuries = pd.read_csv("data/raw/injuries_mock.csv")

    assert len(squads) == 20
    assert len(managers) == 20
    for col_df in (squads, managers, injuries):
        for team in col_df["Team"]:
            assert clean_team_name(team) == team

    start_dates = pd.to_datetime(injuries["StartDate"])
    end_dates = pd.to_datetime(injuries["EndDate"])
    assert (start_dates <= end_dates).all()
    assert ((injuries["Importance"] > 0) & (injuries["Importance"] < 1)).all()


def test_historical_managers_schema_and_coverage():
    path = "data/raw/historical_managers.csv"
    assert os.path.exists(path), f"Missing {path}"
    hist_mgr = pd.read_csv(path)
    assert {"Team", "Manager", "AppointedDate"}.issubset(hist_mgr.columns)
    assert len(hist_mgr) >= 150
    assert hist_mgr["Team"].nunique() >= 35
    pd.to_datetime(hist_mgr["AppointedDate"], format="%Y-%m-%d", errors="raise")
    for team in hist_mgr["Team"]:
        assert clean_team_name(team) == team


def test_sync_live_injuries_from_fpl_bootstrap(tmp_path):
    mock_boot = {
        "teams": [
            {"id": 1, "name": "Arsenal"},
            {"id": 2, "name": "Man Utd"},
            {"id": 3, "name": "Nott'm Forest"},
        ],
        "elements": [
            {
                "id": 10, "first_name": "Bukayo", "second_name": "Saka", "web_name": "Saka",
                "team": 1, "element_type": 4, "status": "i", "chance_of_playing_next_round": 0,
                "news": "Hamstring injury - Expected back 29 Nov", "news_added": "2026-10-25T10:00:00Z",
                "minutes": 900, "total_points": 65, "selected_by_percent": "25.0", "now_cost": 100,
            },
            {
                "id": 11, "first_name": "Luke", "second_name": "Shaw", "web_name": "Shaw",
                "team": 2, "element_type": 2, "status": "d", "chance_of_playing_next_round": 50,
                "news": "Calf injury - 50% chance of playing", "news_added": "2026-11-05T10:00:00Z",
                "minutes": 450, "total_points": 20, "selected_by_percent": "5.0", "now_cost": 50,
            },
            {
                # Non-playing youth academy player should be filtered out
                "id": 12, "first_name": "Academy", "second_name": "Player", "web_name": "Youth",
                "team": 3, "element_type": 3, "status": "i", "chance_of_playing_next_round": 0,
                "news": "Knock - Unknown return date", "news_added": "2026-11-01T10:00:00Z",
                "minutes": 0, "total_points": 0, "selected_by_percent": "0.1", "now_cost": 45,
            },
            {
                # On-loan / unavailable transfer out should be ignored
                "id": 13, "first_name": "Loaned", "second_name": "Out", "web_name": "Loanee",
                "team": 1, "element_type": 3, "status": "u", "chance_of_playing_next_round": 0,
                "news": "Joined another club on loan", "news_added": "2026-09-01T10:00:00Z",
                "minutes": 0, "total_points": 0, "selected_by_percent": "0.1", "now_cost": 50,
            },
        ],
    }
    inj_df = sync_live_injuries(raw_dir=str(tmp_path), bootstrap_json=mock_boot, ref_date="2026-11-11")
    out_csv = tmp_path / "injuries_mock.csv"
    assert out_csv.exists()
    assert len(inj_df) == 2
    assert set(inj_df["Team"]) == {"Arsenal", "Man United"}
    assert ((inj_df["Importance"] > 0) & (inj_df["Importance"] < 1)).all()
    assert (pd.to_datetime(inj_df["StartDate"]) <= pd.to_datetime(inj_df["EndDate"])).all()
    saka = inj_df[inj_df["Player"] == "Bukayo Saka"].iloc[0]
    shaw = inj_df[inj_df["Player"] == "Luke Shaw"].iloc[0]
    assert saka["EndDate"] == "2026-11-29"
    assert saka["Importance"] > shaw["Importance"]


def test_requirements_and_automation_files_present():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        reqs = f.read().lower()
    assert "pytest" in reqs

    expected_files = [
        ".github/workflows/ci.yml",
        ".github/workflows/weekly_refresh.yml",
        "scripts/run_weekly_job.ps1",
        "scripts/register_weekly_task.ps1",
    ]
    for path in expected_files:
        assert os.path.exists(path), f"Missing automation artifact: {path}"


