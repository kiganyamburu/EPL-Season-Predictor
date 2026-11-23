import re
import os
import pandas as pd

def apply_gw11():
    # 1. Update fixtures_raw.csv
    fixtures_path = 'data/raw/fixtures_raw.csv'
    f_df = pd.read_csv(fixtures_path)

    updates = {
        100: ('Aston Villa', 'Sunderland', 0, 0),
        101: ('Bournemouth', "Nott'm Forest", 2, 0),
        102: ('Chelsea', 'Leeds', 1, 4),
        103: ('Coventry', 'Crystal Palace', 3, 2),
        104: ('Man City', 'Fulham', 1, 1),
        105: ('Newcastle', 'Arsenal', 2, 2),
        106: ('Tottenham', 'Ipswich', 2, 0),
        107: ('Hull', 'Brighton', 2, 0),
        108: ('Liverpool', 'Man United', 4, 0),
    }

    for idx, (h, a, hg, ag) in updates.items():
        row = f_df.loc[idx]
        assert row['HomeTeam'] == h and row['AwayTeam'] == a, f"Mismatch at {idx}: {row['HomeTeam']} vs {h}"
        base_sum = re.sub(r'\s*\(\d+-\d+\)', '', str(row['Summary'])).strip()
        f_df.at[idx, 'Summary'] = f"{base_sum} ({hg}-{ag})"
        print(f"Updated fixture {idx}: {f_df.at[idx, 'Summary']}")

    f_df.to_csv(fixtures_path, index=False)
    print("Successfully updated data/raw/fixtures_raw.csv")

    # 2. Update E0_2627.csv
    e0_path = 'data/raw/historical/E0_2627.csv'
    e0_df = pd.read_csv(e0_path)

    gw11_stats = [
        {'Div': 'E0', 'Date': '21/11/2026', 'Time': '15:00', 'HomeTeam': 'Aston Villa', 'AwayTeam': 'Sunderland',
         'FTHG': 0, 'FTAG': 0, 'FTR': 'D', 'HTHG': 0, 'HTAG': 0, 'HTR': 'D', 'Referee': 'M Oliver',
         'HxG': 1.40, 'AxG': 0.70, 'HS': 14, 'AS': 7, 'HST': 4, 'AST': 2, 'HF': 10, 'AF': 11, 'HC': 6, 'AC': 3, 'HY': 1, 'AY': 2, 'HR': 0, 'AR': 0},
        {'Div': 'E0', 'Date': '21/11/2026', 'Time': '15:00', 'HomeTeam': 'Bournemouth', 'AwayTeam': "Nott'm Forest",
         'FTHG': 2, 'FTAG': 0, 'FTR': 'H', 'HTHG': 1, 'HTAG': 0, 'HTR': 'H', 'Referee': 'S Barrott',
         'HxG': 1.57, 'AxG': 0.99, 'HS': 13, 'AS': 9, 'HST': 5, 'AST': 3, 'HF': 11, 'AF': 12, 'HC': 5, 'AC': 4, 'HY': 2, 'AY': 1, 'HR': 0, 'AR': 0},
        {'Div': 'E0', 'Date': '21/11/2026', 'Time': '15:00', 'HomeTeam': 'Chelsea', 'AwayTeam': 'Leeds',
         'FTHG': 1, 'FTAG': 4, 'FTR': 'A', 'HTHG': 1, 'HTAG': 2, 'HTR': 'A', 'Referee': 'A Taylor',
         'HxG': 1.44, 'AxG': 1.91, 'HS': 15, 'AS': 11, 'HST': 4, 'AST': 7, 'HF': 12, 'AF': 14, 'HC': 7, 'AC': 4, 'HY': 2, 'AY': 3, 'HR': 0, 'AR': 0},
        {'Div': 'E0', 'Date': '21/11/2026', 'Time': '15:00', 'HomeTeam': 'Coventry', 'AwayTeam': 'Crystal Palace',
         'FTHG': 3, 'FTAG': 2, 'FTR': 'H', 'HTHG': 1, 'HTAG': 1, 'HTR': 'D', 'Referee': 'T Harrington',
         'HxG': 1.74, 'AxG': 1.57, 'HS': 12, 'AS': 13, 'HST': 6, 'AST': 5, 'HF': 9, 'AF': 10, 'HC': 4, 'AC': 5, 'HY': 1, 'AY': 2, 'HR': 0, 'AR': 0},
        {'Div': 'E0', 'Date': '21/11/2026', 'Time': '15:00', 'HomeTeam': 'Man City', 'AwayTeam': 'Fulham',
         'FTHG': 1, 'FTAG': 1, 'FTR': 'D', 'HTHG': 1, 'HTAG': 0, 'HTR': 'H', 'Referee': 'P Bankes',
         'HxG': 1.98, 'AxG': 0.66, 'HS': 18, 'AS': 6, 'HST': 6, 'AST': 2, 'HF': 8, 'AF': 13, 'HC': 9, 'AC': 2, 'HY': 1, 'AY': 3, 'HR': 0, 'AR': 0},
        {'Div': 'E0', 'Date': '21/11/2026', 'Time': '17:30', 'HomeTeam': 'Newcastle', 'AwayTeam': 'Arsenal',
         'FTHG': 2, 'FTAG': 2, 'FTR': 'D', 'HTHG': 1, 'HTAG': 1, 'HTR': 'D', 'Referee': 'J Brooks',
         'HxG': 1.28, 'AxG': 1.61, 'HS': 11, 'AS': 14, 'HST': 4, 'AST': 5, 'HF': 14, 'AF': 11, 'HC': 5, 'AC': 6, 'HY': 3, 'AY': 2, 'HR': 0, 'AR': 0},
        {'Div': 'E0', 'Date': '21/11/2026', 'Time': '20:00', 'HomeTeam': 'Tottenham', 'AwayTeam': 'Ipswich',
         'FTHG': 2, 'FTAG': 0, 'FTR': 'H', 'HTHG': 1, 'HTAG': 0, 'HTR': 'H', 'Referee': 'R Jones',
         'HxG': 2.11, 'AxG': 0.66, 'HS': 16, 'AS': 6, 'HST': 7, 'AST': 2, 'HF': 10, 'AF': 9, 'HC': 8, 'AC': 2, 'HY': 1, 'AY': 2, 'HR': 0, 'AR': 0},
        {'Div': 'E0', 'Date': '22/11/2026', 'Time': '14:00', 'HomeTeam': 'Hull', 'AwayTeam': 'Brighton',
         'FTHG': 2, 'FTAG': 0, 'FTR': 'H', 'HTHG': 1, 'HTAG': 0, 'HTR': 'H', 'Referee': 'D England',
         'HxG': 1.45, 'AxG': 1.11, 'HS': 10, 'AS': 12, 'HST': 5, 'AST': 3, 'HF': 13, 'AF': 11, 'HC': 3, 'AC': 6, 'HY': 2, 'AY': 2, 'HR': 0, 'AR': 1},
        {'Div': 'E0', 'Date': '22/11/2026', 'Time': '16:30', 'HomeTeam': 'Liverpool', 'AwayTeam': 'Man United',
         'FTHG': 4, 'FTAG': 0, 'FTR': 'H', 'HTHG': 2, 'HTAG': 0, 'HTR': 'H', 'Referee': 'C Kavanagh',
         'HxG': 2.65, 'AxG': 0.74, 'HS': 19, 'AS': 8, 'HST': 9, 'AST': 2, 'HF': 12, 'AF': 10, 'HC': 7, 'AC': 3, 'HY': 1, 'AY': 3, 'HR': 0, 'AR': 1},
    ]

    new_rows_df = pd.DataFrame(gw11_stats)
    existing_keys = set(zip(e0_df['HomeTeam'], e0_df['AwayTeam']))
    new_rows_df = new_rows_df[~new_rows_df.apply(lambda r: (r['HomeTeam'], r['AwayTeam']) in existing_keys, axis=1)]

    if not new_rows_df.empty:
        updated_e0 = pd.concat([e0_df, new_rows_df], ignore_index=True)
        updated_e0.to_csv(e0_path, index=False)
        print(f"Added {len(new_rows_df)} GW11 matches to {e0_path}. Total rows: {len(updated_e0)}")
    else:
        print("GW11 matches already in E0_2627.csv")

if __name__ == '__main__':
    apply_gw11()
