import os
import re
import pandas as pd
import pytest

from src.data_collection import clean_team_name, TEAM_NAME_MAP


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

