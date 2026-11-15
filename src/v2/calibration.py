import os
import numpy as np
import pandas as pd


def build_prematch_calibration_df(history_df, completed_df):
    """
    Pair each completed match with its latest pre-match prediction from predictions_history.csv.
    Returns a DataFrame with per-match pre-match probabilities, Brier score, log-loss, and outcome hit.
    """
    if history_df is None or history_df.empty or completed_df is None or completed_df.empty:
        return pd.DataFrame()

    hist = history_df.copy()
    if 'Status' in hist.columns:
        pre_match = hist[hist['Status'].fillna('') != 'Completed'].copy()
    else:
        pre_match = hist.copy()

    if pre_match.empty:
        pre_match = hist.copy()

    pre_match['CompletedMatchesCount'] = pd.to_numeric(pre_match.get('CompletedMatchesCount', 0), errors='coerce').fillna(0)
    pre_match = pre_match.sort_values(['CompletedMatchesCount', 'SnapshotDate'])
    latest_pre = pre_match.drop_duplicates(subset=['HomeTeam', 'AwayTeam'], keep='last')

    c_df = completed_df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].copy()
    if 'MatchDay' in history_df.columns:
        md_map = hist.drop_duplicates(subset=['HomeTeam', 'AwayTeam'])[['HomeTeam', 'AwayTeam', 'MatchDay']]
        c_df = c_df.merge(md_map, on=['HomeTeam', 'AwayTeam'], how='left')
    else:
        c_df['Date_dt'] = pd.to_datetime(c_df['Date'])
        c_df = c_df.sort_values('Date_dt').reset_index(drop=True)
        c_df['MatchDay'] = (c_df.index // 10) + 1

    merged = c_df.merge(
        latest_pre[['HomeTeam', 'AwayTeam', 'HomeWin%', 'Draw%', 'AwayWin%', 'MostLikelyScore', 'SnapshotDate', 'CompletedMatchesCount']],
        on=['HomeTeam', 'AwayTeam'],
        how='inner'
    )
    if merged.empty:
        return pd.DataFrame()

    probs = merged[['HomeWin%', 'Draw%', 'AwayWin%']].to_numpy(dtype=float) / 100.0
    row_sums = np.clip(probs.sum(axis=1, keepdims=True), 1e-8, None)
    probs = probs / row_sums

    ftr = merged['FTR'].to_numpy()
    y_oh = np.zeros_like(probs)
    y_oh[ftr == 'H', 0] = 1.0
    y_oh[ftr == 'D', 1] = 1.0
    y_oh[ftr == 'A', 2] = 1.0

    # Multi-class Brier score per match (mean squared error across H/D/A classes)
    brier_per_match = np.sum((probs - y_oh) ** 2, axis=1) / 3.0
    p_actual = np.clip((probs * y_oh).sum(axis=1), 1e-8, 1.0)
    logloss_per_match = -np.log(p_actual)

    pred_idx = np.argmax(probs, axis=1)
    idx_to_res = np.array(['H', 'D', 'A'])
    pred_ftr = idx_to_res[pred_idx]

    merged['pH'] = probs[:, 0]
    merged['pD'] = probs[:, 1]
    merged['pA'] = probs[:, 2]
    merged['PredFTR'] = pred_ftr
    merged['Correct'] = (pred_ftr == ftr).astype(int)
    merged['Brier'] = brier_per_match
    merged['LogLoss'] = logloss_per_match
    merged['MatchDay'] = pd.to_numeric(merged['MatchDay'], errors='coerce').fillna(1).astype(int)
    return merged.sort_values(['MatchDay', 'Date']).reset_index(drop=True)


def compute_calibration_curve(calib_df, n_bins=6):
    """
    Compute reliability curve bins across all predicted (Home/Draw/Away) probabilities.
    Returns DataFrame with BinLabel, MeanPredicted%, ObservedFreq%, Count.
    """
    if calib_df is None or calib_df.empty:
        return pd.DataFrame()

    probs_flat = np.concatenate([calib_df['pH'].to_numpy(), calib_df['pD'].to_numpy(), calib_df['pA'].to_numpy()])
    actual_flat = np.concatenate([
        (calib_df['FTR'] == 'H').to_numpy(dtype=float),
        (calib_df['FTR'] == 'D').to_numpy(dtype=float),
        (calib_df['FTR'] == 'A').to_numpy(dtype=float),
    ])

    edges = np.linspace(0.0, 0.90, n_bins + 1)
    edges[-1] = 1.01
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs_flat >= lo) & (probs_flat < hi)
        if mask.any():
            rows.append({
                'Bin': f"{int(lo * 100)}-{int(min(hi, 1.0) * 100)}%",
                'MeanPredictedPct': round(float(probs_flat[mask].mean() * 100), 2),
                'ObservedFreqPct': round(float(actual_flat[mask].mean() * 100), 2),
                'Count': int(mask.sum()),
            })
    return pd.DataFrame(rows)


def compute_snapshot_title_trajectories(history_df, completed_df, num_sims=2000, seed=42):
    """
    Reconstruct Expected Points and Title Win % trajectories across distinct snapshots
    in predictions_history.csv using fast vectorized multinomial match outcome draws.
    """
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    hist = history_df.copy()
    hist['CompletedMatchesCount'] = pd.to_numeric(hist.get('CompletedMatchesCount', 0), errors='coerce').fillna(0).astype(int)
    latest_dates = (
        hist.groupby('CompletedMatchesCount')['SnapshotDate']
        .max()
        .reset_index()
        .sort_values('CompletedMatchesCount')
    )

    c_sorted = pd.DataFrame()
    if completed_df is not None and not completed_df.empty:
        c_sorted = completed_df.copy()
        c_sorted['Date_dt'] = pd.to_datetime(c_sorted['Date'])
        c_sorted = c_sorted.sort_values('Date_dt').reset_index(drop=True)

    teams = sorted(list(set(hist['HomeTeam']).union(set(hist['AwayTeam']))))
    team_to_idx = {t: i for i, t in enumerate(teams)}
    num_teams = len(teams)
    rng = np.random.default_rng(seed)

    trajectory_rows = []
    for _, snap_meta in latest_dates.iterrows():
        cnt = int(snap_meta['CompletedMatchesCount'])
        sdate = snap_meta['SnapshotDate']
        snap = hist[(hist['CompletedMatchesCount'] == cnt) & (hist['SnapshotDate'] == sdate)].copy()
        if snap.empty:
            continue

        # Base points & GD from completed matches recorded up to `cnt` when snapshot lacked earlier completed rows
        base_pts = np.zeros(num_teams, dtype=float)
        base_gd = np.zeros(num_teams, dtype=float)
        completed_keys = set()

        if 'Status' in snap.columns and (snap['Status'] == 'Completed').any():
            for _, r in snap[snap['Status'] == 'Completed'].iterrows():
                score = str(r.get('ActualScore', ''))
                if '-' in score:
                    try:
                        hg, ag = map(int, score.split('-'))
                        hi, ai = team_to_idx[r['HomeTeam']], team_to_idx[r['AwayTeam']]
                        completed_keys.add((r['HomeTeam'], r['AwayTeam']))
                        base_pts[hi] += 3 if hg > ag else (1 if hg == ag else 0)
                        base_pts[ai] += 3 if ag > hg else (1 if hg == ag else 0)
                        base_gd[hi] += hg - ag
                        base_gd[ai] += ag - hg
                    except ValueError:
                        pass

        if len(completed_keys) < cnt and not c_sorted.empty:
            for _, r in c_sorted.head(cnt).iterrows():
                key = (r['HomeTeam'], r['AwayTeam'])
                if key in completed_keys or r['HomeTeam'] not in team_to_idx or r['AwayTeam'] not in team_to_idx:
                    continue
                completed_keys.add(key)
                hg, ag = int(r['FTHG']), int(r['FTAG'])
                hi, ai = team_to_idx[r['HomeTeam']], team_to_idx[r['AwayTeam']]
                base_pts[hi] += 3 if hg > ag else (1 if hg == ag else 0)
                base_pts[ai] += 3 if ag > hg else (1 if hg == ag else 0)
                base_gd[hi] += hg - ag
                base_gd[ai] += ag - hg

        rem = snap[~snap.apply(lambda r: (r['HomeTeam'], r['AwayTeam']) in completed_keys, axis=1)].copy()

        pts_sims = np.tile(base_pts, (num_sims, 1))
        gd_sims = np.tile(base_gd, (num_sims, 1))

        if not rem.empty:
            probs = rem[['HomeWin%', 'Draw%', 'AwayWin%']].to_numpy(dtype=float) / 100.0
            probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-8, None)
            cdf_h = probs[:, 0]
            cdf_hd = probs[:, 0] + probs[:, 1]
            u = rng.random((num_sims, len(rem)))

            h_idxs = np.array([team_to_idx[t] for t in rem['HomeTeam']], dtype=int)
            a_idxs = np.array([team_to_idx[t] for t in rem['AwayTeam']], dtype=int)

            is_h = u < cdf_h[None, :]
            is_d = (u >= cdf_h[None, :]) & (u < cdf_hd[None, :])
            is_a = u >= cdf_hd[None, :]

            h_pts_mat = np.where(is_h, 3.0, np.where(is_d, 1.0, 0.0))
            a_pts_mat = np.where(is_a, 3.0, np.where(is_d, 1.0, 0.0))
            gd_mat = np.where(is_h, 1.4, np.where(is_a, -1.4, 0.0))

            for m_i in range(len(rem)):
                hi, ai = h_idxs[m_i], a_idxs[m_i]
                pts_sims[:, hi] += h_pts_mat[:, m_i]
                pts_sims[:, ai] += a_pts_mat[:, m_i]
                gd_sims[:, hi] += gd_mat[:, m_i]
                gd_sims[:, ai] -= gd_mat[:, m_i]

        # Determine champion & top 4 per simulation
        ranks = np.argsort(-(pts_sims + gd_sims * 1e-4), axis=1)
        champions = ranks[:, 0]
        label = f"GW{cnt // 10} ({cnt} played)" if cnt > 0 else "Pre-Season (0 played)"

        for t_idx, team in enumerate(teams):
            trajectory_rows.append({
                'CompletedMatchesCount': cnt,
                'SnapshotLabel': label,
                'SnapshotDate': sdate,
                'Team': team,
                'ExpectedPoints': round(float(np.median(pts_sims[:, t_idx])), 1),
                'TitleWin%': round(float(np.mean(champions == t_idx) * 100.0), 2),
            })

    return pd.DataFrame(trajectory_rows)

