import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="EPL Season Predictor 2026/27",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Theme with Glowing Accents)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
}

/* Main title styling with gradient */
.main-title {
    font-size: 3rem;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    text-shadow: 0 4px 12px rgba(0, 242, 254, 0.15);
}

.subtitle {
    font-size: 1.2rem;
    color: #94a3b8;
    margin-bottom: 2rem;
}

/* Glassmorphism KPI card */
.kpi-card {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    text-align: center;
    transition: transform 0.3s ease, border-color 0.3s ease;
}

.kpi-card:hover {
    transform: translateY(-5px);
    border-color: rgba(0, 242, 254, 0.3);
}

.kpi-val {
    font-size: 2.5rem;
    font-weight: 800;
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}

.kpi-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 1.5px;
}
</style>
""", unsafe_allow_html=True)

# Helper function to load simulated and live data
def load_data():
    preds_path = 'output/predictions.csv'
    standings_path = 'output/final_table_projection.csv'
    squad_path = 'data/raw/squad_values.csv'
    transfers_path = 'data/raw/transfers.csv'
    hist_path = 'data/processed/historical_results.csv'
    
    preds_df = pd.read_csv(preds_path) if os.path.exists(preds_path) else None
    standings_df = pd.read_csv(standings_path) if os.path.exists(standings_path) else None
    squad_df = pd.read_csv(squad_path) if os.path.exists(squad_path) else None
    transfers_df = pd.read_csv(transfers_path) if os.path.exists(transfers_path) else None
    hist_df = pd.read_csv(hist_path) if os.path.exists(hist_path) else None
    
    # Compute live standings from fixtures_raw.csv (matches played with actual scorelines)
    from src.data_collection import parse_completed_fixtures_from_raw
    completed_df, unplayed_df = parse_completed_fixtures_from_raw('data/raw')
    
    live_df = None
    if not completed_df.empty:
        teams = sorted(list(set(standings_df['Team']) if standings_df is not None else set(completed_df['HomeTeam'])))
        stats = {t: {'Team': t, 'P': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0, 'Form': []} for t in teams}
        
        c_df = completed_df.copy()
        c_df['Date'] = pd.to_datetime(c_df['Date'])
        c_df = c_df.sort_values('Date').reset_index(drop=True)
        
        for _, row in c_df.iterrows():
            h = row['HomeTeam']
            a = row['AwayTeam']
            hg = int(row['FTHG'])
            ag = int(row['FTAG'])
            
            if h in stats and a in stats:
                stats[h]['P'] += 1
                stats[a]['P'] += 1
                stats[h]['GF'] += hg
                stats[h]['GA'] += ag
                stats[a]['GF'] += ag
                stats[a]['GA'] += hg
                stats[h]['GD'] += (hg - ag)
                stats[a]['GD'] += (ag - hg)
                
                if hg > ag:
                    stats[h]['W'] += 1
                    stats[h]['Pts'] += 3
                    stats[h]['Form'].append('W')
                    stats[a]['L'] += 1
                    stats[a]['Form'].append('L')
                elif ag > hg:
                    stats[a]['W'] += 1
                    stats[a]['Pts'] += 3
                    stats[a]['Form'].append('W')
                    stats[h]['L'] += 1
                    stats[h]['Form'].append('L')
                else:
                    stats[h]['D'] += 1
                    stats[h]['Pts'] += 1
                    stats[h]['Form'].append('D')
                    stats[a]['D'] += 1
                    stats[a]['Pts'] += 1
                    stats[a]['Form'].append('D')
                    
        for t in stats:
            recent = stats[t]['Form'][-6:]
            stats[t]['RecentForm'] = " ".join(recent) if recent else "-"
            
        live_df = pd.DataFrame(list(stats.values()))
        live_df = live_df.sort_values(by=['Pts', 'GD', 'GF'], ascending=[False, False, False]).reset_index(drop=True)
        live_df['Rank'] = live_df.index + 1
        
    return preds_df, standings_df, squad_df, transfers_df, hist_df, live_df, completed_df

preds_df, standings_df, squad_df, transfers_df, hist_df, live_df, completed_df = load_data()

# Check if data exists
if standings_df is None or preds_df is None:
    st.error("Error: Simulation outputs not found in `/output/`. Please run the pipeline script first using the terminal command: `python -m src.pipeline_v2`")
    st.stop()

# Header Section
st.markdown('<div class="main-title">EPL Season Predictor 2026/27</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">10,000 Monte Carlo Season Simulations Blending Dixon-Coles Poisson & XGBoost Models</div>', unsafe_allow_html=True)

# Top KPIs
col1, col2, col3 = st.columns(3)

# Find top title winner
title_fav = standings_df.iloc[0]
top4_fav = standings_df.sort_values('Top4%', ascending=False).iloc[0]
relegation_danger = standings_df.sort_values('Relegated%', ascending=False).iloc[0]

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-val">{title_fav['Team']}</div>
        <div class="kpi-label"> Title Favorite ({title_fav['TitleWin%']:.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-val">{top4_fav['Team']}</div>
        <div class="kpi-label"> Top 4 Lock ({top4_fav['Top4%']:.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-val">{relegation_danger['Team']}</div>
        <div class="kpi-label">⚠️ Relegation Danger ({relegation_danger['Relegated%']:.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Tabs Setup
tab1, tab2, tab3, tab_h2h, tab_transfers, tab4, tab5 = st.tabs([
    "📊 Standings (Live & Projected)", 
    "📈 Standing Distributions", 
    "⚽ Match Predictions", 
    "⚔️ Head-to-Head Simulator",
    "🔄 Squad Transfers & Signings",
    "🔍 Narrative Insights",
    "🔮 What-If Sandbox"
])

# ------------------------------------------------------------------------------
# TAB 1: LEAGUE STANDINGS (LIVE, PROJECTED & COMPARISON)
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Premier League Standings")
    
    col_t_mode, col_space = st.columns([3, 1])
    with col_t_mode:
        table_mode = st.radio(
            "Standings View Mode:",
            ["📊 Projected Final Table", "🏆 Current Live Standings (Played So Far)", "⚖️ Side-by-Side (Current vs. Projected)"],
            horizontal=True,
            key="standings_view_mode"
        )
    
    if table_mode == "📊 Projected Final Table":
        st.write("This table shows the median final points, goal difference, and ranks across 10,000 simulations, together with 90% confidence intervals.")
        display_df = standings_df.copy()
        display_df.columns = [
            'Team', 'Exp Points', 'Points Range (90% CI)', 'Exp GD', 
            'GD Range', 'Exp Rank', 'Rank Range', 'Title Win %', 'Top 4 %', 'Relegation %'
        ]
        styler = display_df.style.background_gradient(
            subset=['Title Win %', 'Top 4 %'], cmap='Blues'
        ).background_gradient(
            subset=['Relegation %'], cmap='Reds'
        ).format({
            'Title Win %': '{:.2f}%',
            'Top 4 %': '{:.2f}%',
            'Relegation %': '{:.2f}%'
        })
        st.dataframe(styler, use_container_width=True, height=730)
        
    elif table_mode == "🏆 Current Live Standings (Played So Far)":
        if live_df is not None and not live_df.empty:
            matches_count = len(completed_df) if completed_df is not None else 0
            st.write(f"Actual Premier League table based on **{matches_count} completed matches** recorded in the 2026/27 campaign.")
            
            # KPI Cards
            col_k1, col_k2, col_k3, col_k4 = st.columns(4)
            with col_k1:
                st.markdown(f"""
                <div class="kpi-card" style="padding: 16px;">
                    <div style="font-size: 1.8rem; font-weight: 800; color: #00f2fe;">{matches_count} / 380</div>
                    <div class="kpi-label">Matches Completed</div>
                </div>
                """, unsafe_allow_html=True)
            with col_k2:
                leader = live_df.iloc[0]
                st.markdown(f"""
                <div class="kpi-card" style="padding: 16px;">
                    <div style="font-size: 1.8rem; font-weight: 800; color: #10b981;">{leader['Team']}</div>
                    <div class="kpi-label">Current Leader ({leader['Pts']} pts)</div>
                </div>
                """, unsafe_allow_html=True)
            with col_k3:
                top_gf = live_df.sort_values('GF', ascending=False).iloc[0]
                st.markdown(f"""
                <div class="kpi-card" style="padding: 16px;">
                    <div style="font-size: 1.8rem; font-weight: 800; color: #38bdf8;">{top_gf['Team']}</div>
                    <div class="kpi-label">Best Attack ({top_gf['GF']} goals)</div>
                </div>
                """, unsafe_allow_html=True)
            with col_k4:
                top_ga = live_df.sort_values('GA', ascending=True).iloc[0]
                st.markdown(f"""
                <div class="kpi-card" style="padding: 16px;">
                    <div style="font-size: 1.8rem; font-weight: 800; color: #a855f7;">{top_ga['Team']}</div>
                    <div class="kpi-label">Best Defense ({top_ga['GA']} conceded)</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            display_live = live_df[['Rank', 'Team', 'P', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'RecentForm']].copy()
            display_live = display_live.rename(columns={'RecentForm': 'Form (Last 6)'})
            
            styler_live = display_live.style.background_gradient(
                subset=['Pts', 'GD'], cmap='Blues'
            )
            st.dataframe(styler_live, use_container_width=True, height=730, hide_index=True)
        else:
            st.info("No completed match results recorded in the fixtures feed yet.")
            
    else:  # Side-by-Side (Current vs. Projected)
        if live_df is not None and standings_df is not None:
            st.write("Compare current positions against final projected finishes to identify overperformers and comeback candidates.")
            
            comp = pd.merge(
                live_df[['Rank', 'Team', 'P', 'Pts', 'GD']],
                standings_df[['Team', 'ExpectedRank', 'ExpectedPoints', 'ExpectedGD', 'TitleWin%', 'Top4%', 'Relegated%']],
                on='Team'
            )
            comp = comp.rename(columns={
                'Rank': 'Live Rank',
                'Pts': 'Live Pts',
                'ExpectedRank': 'Proj Rank',
                'ExpectedPoints': 'Proj Pts',
                'ExpectedGD': 'Proj GD'
            })
            comp['Rank Delta'] = comp['Live Rank'] - comp['Proj Rank']
            def format_delta(d):
                if d > 0:
                    return f"▲ +{d} (Climb)"
                elif d < 0:
                    return f"▼ {d} (Fall)"
                return "= (Steady)"
            comp['Trajectory'] = comp['Rank Delta'].apply(format_delta)
            comp['Pts to Add'] = comp['Proj Pts'] - comp['Live Pts']
            comp = comp.sort_values('Live Rank').reset_index(drop=True)
            
            show_cols = ['Live Rank', 'Team', 'P', 'Live Pts', 'Proj Pts', 'Pts to Add', 'Proj Rank', 'Trajectory', 'TitleWin%', 'Top4%', 'Relegated%']
            st.dataframe(
                comp[show_cols].style.background_gradient(
                    subset=['Live Pts', 'Proj Pts'], cmap='Blues'
                ).background_gradient(
                    subset=['TitleWin%', 'Top4%'], cmap='Greens'
                ).background_gradient(
                    subset=['Relegated%'], cmap='Reds'
                ).format({
                    'Live Pts': '{:.0f}',
                    'Proj Pts': '{:.1f}',
                    'Pts to Add': '{:+.1f}',
                    'TitleWin%': '{:.1f}%',
                    'Top4%': '{:.1f}%',
                    'Relegated%': '{:.1f}%'
                }),
                use_container_width=True,
                height=730,
                hide_index=True
            )
            
            st.write("---")
            st.subheader("📈 Current Rank vs. Projected Final Position")
            fig_comp = px.scatter(
                comp,
                x='Live Rank',
                y='Proj Rank',
                text='Team',
                size='TitleWin%',
                color='Rank Delta',
                color_continuous_scale='RdYlGn',
                labels={'Live Rank': 'Current Position', 'Proj Rank': 'Projected Final Position'}
            )
            fig_comp.add_shape(
                type="line", line=dict(dash="dash", color="rgba(255,255,255,0.25)"),
                x0=1, y0=1, x1=20, y1=20
            )
            fig_comp.update_traces(textposition='top center')
            fig_comp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', dtick=1, autorange="reversed"),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', dtick=1, autorange="reversed"),
                height=550,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig_comp, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: STANDING DISTRIBUTIONS (CHARTS)
# ------------------------------------------------------------------------------
with tab2:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Expected Points Range per Team")
        
        # Parse points range for error bars
        # points range format: "MIN-MAX"
        pts_min = []
        pts_max = []
        for r in standings_df['PointsRange']:
            mn, mx = map(float, r.split('-'))
            pts_min.append(mn)
            pts_max.append(mx)
            
        standings_df['PointsMin'] = pts_min
        standings_df['PointsMax'] = pts_max
        standings_df['ErrorMinus'] = standings_df['ExpectedPoints'] - standings_df['PointsMin']
        standings_df['ErrorPlus'] = standings_df['PointsMax'] - standings_df['ExpectedPoints']
        
        # Bar chart with error bars
        fig_pts = go.Figure()
        fig_pts.add_trace(go.Bar(
            name='Expected Points',
            x=standings_df['Team'],
            y=standings_df['ExpectedPoints'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=standings_df['ErrorPlus'],
                arrayminus=standings_df['ErrorMinus'],
                color='#ef4444'
            ),
            marker=dict(
                color=standings_df['ExpectedPoints'],
                colorscale='Viridis',
                line=dict(width=1, color='rgba(255,255,255,0.1)')
            )
        ))
        
        fig_pts.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickangle=-45),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Points"),
            height=500,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig_pts, use_container_width=True)
        
    with col_chart2:
        st.subheader("Title Win Probability (%)")
        
        title_teams = standings_df[standings_df['TitleWin%'] > 0.1].sort_values('TitleWin%', ascending=True)
        
        fig_title = px.bar(
            title_teams,
            y='Team',
            x='TitleWin%',
            orientation='h',
            color='TitleWin%',
            color_continuous_scale='teal',
            labels={'TitleWin%': 'Title Probability (%)'}
        )
        
        fig_title.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            coloraxis_showscale=False,
            height=500,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig_title, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: MATCH PREDICTIONS
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Match-by-Match Probabilities")
    st.write("Browse win/draw/loss odds and the most likely scoreline for all 380 upcoming fixtures.")
    
    # Filters
    col_f1, col_f2, col_f3 = st.columns([1, 1, 1.5])
    with col_f1:
        matchday_filter = st.selectbox(
            "Matchday", 
            ["All"] + sorted(list(preds_df['MatchDay'].unique()))
        )
    with col_f2:
        status_options = ["All"]
        if 'Status' in preds_df.columns:
            status_options += sorted(list(preds_df['Status'].dropna().unique()))
        status_filter = st.selectbox("Status", status_options)
    with col_f3:
        team_search = st.text_input("Filter by Team Name").strip()
        
    # Apply filters
    filtered_preds = preds_df.copy()
    if matchday_filter != "All":
        filtered_preds = filtered_preds[filtered_preds['MatchDay'] == matchday_filter]
        
    if status_filter != "All" and 'Status' in filtered_preds.columns:
        filtered_preds = filtered_preds[filtered_preds['Status'] == status_filter]
        
    if team_search:
        filtered_preds = filtered_preds[
            filtered_preds['HomeTeam'].str.contains(team_search, case=False) | 
            filtered_preds['AwayTeam'].str.contains(team_search, case=False)
        ]
        
    # Render table or cards
    if len(filtered_preds) == 0:
        st.info("No matches found matching the criteria.")
    else:
        # Display as a styled table
        display_preds = filtered_preds.copy()
        display_preds = display_preds.rename(columns={
            'MatchDay': 'GW',
            'HomeTeam': 'Home Team',
            'AwayTeam': 'Away Team',
            'ActualScore': 'Actual Score',
            'HomeWin%': 'Home Win %',
            'Draw%': 'Draw %',
            'AwayWin%': 'Away Win %',
            'MostLikelyScore': 'Predicted / Final Score'
        })
        
        # Select columns to display
        pref_cols = ['GW', 'Date', 'Home Team', 'Away Team', 'Status', 'Actual Score', 'Home Win %', 'Draw %', 'Away Win %', 'Predicted / Final Score']
        cols_to_show = [c for c in pref_cols if c in display_preds.columns]
        
        st.dataframe(
            display_preds[cols_to_show].style.format({
                'Home Win %': '{:.1f}%',
                'Draw %': '{:.1f}%',
                'Away Win %': '{:.1f}%',
                'Actual Score': lambda v: '-' if pd.isna(v) or v is None else str(v)
            }), 
            use_container_width=True, 
            height=430,
            hide_index=True
        )
        
        # Interactive Match Detail card showing explainability breakdown
        st.write("---")
        st.subheader("🔍 Match Prediction Driver Inspector")
        st.write("Select any match from the filtered list below to view the human-readable explanation generated by the explainability module.")
        
        fixture_options = filtered_preds.index.tolist()
        if fixture_options:
            def format_option(i):
                r = filtered_preds.loc[i]
                status_badge = f"[{r['Status']}] " if 'Status' in r and pd.notna(r['Status']) else ""
                score_badge = f" ({r['ActualScore']})" if 'ActualScore' in r and pd.notna(r['ActualScore']) else ""
                return f"{status_badge}GW {r['MatchDay']}: {r['HomeTeam']} vs. {r['AwayTeam']}{score_badge}"
                
            selected_idx = st.selectbox(
                "Select Match to Inspect:", 
                fixture_options,
                format_func=format_option
            )
            
            match_row = filtered_preds.loc[selected_idx]
            
            if 'Status' in match_row and match_row['Status'] == 'Completed':
                st.success(f"⚽ **Completed Match** | Score: **{match_row.get('ActualScore', '-')}** | Date: **{match_row['Date']}**")
            
            col_o1, col_o2, col_o3 = st.columns(3)
            with col_o1:
                st.metric(label=f"🏠 {match_row['HomeTeam']} Win %", value=f"{match_row['HomeWin%']:.1f}%")
            with col_o2:
                st.metric(label="🤝 Draw %", value=f"{match_row['Draw%']:.1f}%")
            with col_o3:
                st.metric(label=f"✈️ {match_row['AwayTeam']} Win %", value=f"{match_row['AwayWin%']:.1f}%")
                
            # Render explanation
            explanation_text = match_row.get('Explanation', 'No explanation generated for this match.')
            st.info(f"💡 **Model Factors & Explanations**:\n\n{explanation_text}")


# ------------------------------------------------------------------------------
# TAB H2H: HEAD-TO-HEAD SIMULATOR & SCORE HEATMAP
# ------------------------------------------------------------------------------
with tab_h2h:
    st.subheader("⚔️ Head-to-Head Matchup Simulator")
    st.write("Simulate any Premier League fixture to analyze score probability heatmaps, derived betting market odds, and historical rivalry records.")
    
    active_teams = sorted(list(standings_df['Team'].unique()))
    
    col_sel1, col_swap, col_sel2 = st.columns([5, 1, 5])
    with col_sel1:
        default_h_idx = active_teams.index('Arsenal') if 'Arsenal' in active_teams else 0
        h2h_home = st.selectbox("🏠 Select Home Team", active_teams, index=default_h_idx, key="h2h_home_team")
    with col_swap:
        st.write("")
        st.write("")
        st.markdown("<div style='text-align: center; font-size: 1.5rem; font-weight: 800; color: #00f2fe; padding-top: 6px;'>VS</div>", unsafe_allow_html=True)
    with col_sel2:
        default_a_idx = active_teams.index('Man City') if 'Man City' in active_teams else 1
        h2h_away = st.selectbox("✈️ Select Away Team", active_teams, index=default_a_idx, key="h2h_away_team")
        
    if h2h_home == h2h_away:
        st.warning("⚠️ Please select two different teams to run the head-to-head matchup simulation.")
    else:
        # Load final Elo ratings
        final_elos_map = {}
        if os.path.exists('data/processed/final_elos.csv'):
            f_elo_df = pd.read_csv('data/processed/final_elos.csv')
            final_elos_map = dict(zip(f_elo_df['Team'], f_elo_df['Elo']))
            
        h_elo = final_elos_map.get(h2h_home, 1500.0)
        a_elo = final_elos_map.get(h2h_away, 1500.0)
        elo_diff = h_elo - a_elo
        
        # Check if match is in precomputed schedule
        match_pred = None
        if preds_df is not None:
            matches_found = preds_df[(preds_df['HomeTeam'] == h2h_home) & (preds_df['AwayTeam'] == h2h_away)]
            if not matches_found.empty:
                match_pred = matches_found.iloc[0]
                
        # Expected goals (Home advantage ~ 1.25)
        home_exp_g = max(0.25, 1.35 + elo_diff / 450.0)
        away_exp_g = max(0.20, 1.15 - elo_diff / 450.0)
        
        # Compute joint Poisson distribution with Dixon-Coles low-score correlation
        max_g = 5
        score_matrix = np.zeros((max_g + 1, max_g + 1))
        from scipy.stats import poisson
        rho = -0.015
        for x in range(max_g + 1):
            for y in range(max_g + 1):
                px = poisson.pmf(x, home_exp_g)
                py = poisson.pmf(y, away_exp_g)
                adj = 1.0
                if x == 0 and y == 0:
                    adj = 1.0 - home_exp_g * away_exp_g * rho
                elif x == 1 and y == 0:
                    adj = 1.0 + away_exp_g * rho
                elif x == 0 and y == 1:
                    adj = 1.0 + home_exp_g * rho
                elif x == 1 and y == 1:
                    adj = 1.0 - rho
                adj = max(0.01, adj)
                score_matrix[x, y] = px * py * adj
        score_matrix /= score_matrix.sum()
        
        # Calculate outcome probabilities
        p_home_win = np.sum(np.triu(score_matrix, 1).T)
        p_draw = np.sum(np.diag(score_matrix))
        p_away_win = np.sum(np.tril(score_matrix, -1).T)
        
        # Align with precomputed ensemble predictions if available
        if match_pred is not None:
            p_home_win = match_pred['HomeWin%'] / 100.0
            p_draw = match_pred['Draw%'] / 100.0
            p_away_win = match_pred['AwayWin%'] / 100.0
            
            # Rescale quadrants of score matrix to perfectly match blended ensemble
            sum_h = np.sum(np.triu(score_matrix, 1).T)
            sum_d = np.sum(np.diag(score_matrix))
            sum_a = np.sum(np.tril(score_matrix, -1).T)
            for x in range(max_g + 1):
                for y in range(max_g + 1):
                    if x > y and sum_h > 0:
                        score_matrix[x, y] *= p_home_win / sum_h
                    elif x < y and sum_a > 0:
                        score_matrix[x, y] *= p_away_win / sum_a
                    elif x == y and sum_d > 0:
                        score_matrix[x, y] *= p_draw / sum_d
            score_matrix /= score_matrix.sum()
            
        # Top Probability KPI Cards
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.markdown(f"""
            <div class="kpi-card" style="padding: 16px; border-color: rgba(16, 185, 129, 0.4);">
                <div style="font-size: 2.2rem; font-weight: 800; color: #10b981;">{p_home_win*100:.1f}%</div>
                <div class="kpi-label">🏠 {h2h_home} Win</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Fair Odds: <strong>{1/max(0.01, p_home_win):.2f}</strong></div>
            </div>
            """, unsafe_allow_html=True)
        with col_res2:
            st.markdown(f"""
            <div class="kpi-card" style="padding: 16px; border-color: rgba(245, 158, 11, 0.4);">
                <div style="font-size: 2.2rem; font-weight: 800; color: #f59e0b;">{p_draw*100:.1f}%</div>
                <div class="kpi-label">🤝 Draw</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Fair Odds: <strong>{1/max(0.01, p_draw):.2f}</strong></div>
            </div>
            """, unsafe_allow_html=True)
        with col_res3:
            st.markdown(f"""
            <div class="kpi-card" style="padding: 16px; border-color: rgba(56, 189, 248, 0.4);">
                <div style="font-size: 2.2rem; font-weight: 800; color: #38bdf8;">{p_away_win*100:.1f}%</div>
                <div class="kpi-label">✈️ {h2h_away} Win</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Fair Odds: <strong>{1/max(0.01, p_away_win):.2f}</strong></div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        
        # Most likely score
        max_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        ml_score_str = f"{max_idx[0]} - {max_idx[1]}"
        ml_prob = score_matrix[max_idx] * 100.0
        
        # Columns: Score Heatmap (Left) and Derived Markets (Right)
        col_heat, col_mkts = st.columns([6, 5])
        
        with col_heat:
            st.markdown(f"### 🎯 Score Probability Heatmap")
            st.caption(f"Most Likely Score: **{ml_score_str}** ({ml_prob:.1f}% probability) | Exp Goals: **{home_exp_g:.2f}** vs **{away_exp_g:.2f}**")
            
            z_vals = np.round(score_matrix * 100, 1)
            y_labels = [f"{h2h_home}: {i}" for i in range(max_g + 1)]
            x_labels = [f"{h2h_away}: {i}" for i in range(max_g + 1)]
            
            fig_heat = go.Figure(data=go.Heatmap(
                z=z_vals,
                x=x_labels,
                y=y_labels,
                text=np.array([[f"{z_vals[r, c]:.1f}%" for c in range(max_g + 1)] for r in range(max_g + 1)]),
                texttemplate="%{text}",
                textfont=dict(size=12, color="white"),
                colorscale="Viridis",
                showscale=False
            ))
            fig_heat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=f"{h2h_away} Goals"),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=f"{h2h_home} Goals", autorange="reversed"),
                height=420,
                margin=dict(l=40, r=40, t=10, b=10)
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
        with col_mkts:
            st.markdown("### 📊 Derived Goal & Betting Markets")
            
            # Goal totals
            goals_sum = np.add.outer(np.arange(max_g + 1), np.arange(max_g + 1))
            p_o15 = np.sum(score_matrix[goals_sum >= 2])
            p_u15 = max(0.001, 1.0 - p_o15)
            p_o25 = np.sum(score_matrix[goals_sum >= 3])
            p_u25 = max(0.001, 1.0 - p_o25)
            p_o35 = np.sum(score_matrix[goals_sum >= 4])
            p_u35 = max(0.001, 1.0 - p_o35)
            
            # Both teams to score
            p_btts_yes = np.sum(score_matrix[1:, 1:])
            p_btts_no = max(0.001, 1.0 - p_btts_yes)
            
            # Clean sheets
            p_cs_home = np.sum(score_matrix[:, 0]) # Away scores 0
            p_cs_away = np.sum(score_matrix[0, :]) # Home scores 0
            
            # Double chance
            p_1x = p_home_win + p_draw
            p_x2 = p_draw + p_away_win
            
            mkt_data = [
                {"Market": "Over 1.5 Goals", "Probability": f"{p_o15*100:.1f}%", "Fair Odds": f"{1/p_o15:.2f}", "Alternative": "Under 1.5", "Alt Prob": f"{p_u15*100:.1f}%", "Alt Odds": f"{1/p_u15:.2f}"},
                {"Market": "Over 2.5 Goals", "Probability": f"{p_o25*100:.1f}%", "Fair Odds": f"{1/p_o25:.2f}", "Alternative": "Under 2.5", "Alt Prob": f"{p_u25*100:.1f}%", "Alt Odds": f"{1/p_u25:.2f}"},
                {"Market": "Over 3.5 Goals", "Probability": f"{p_o35*100:.1f}%", "Fair Odds": f"{1/p_o35:.2f}", "Alternative": "Under 3.5", "Alt Prob": f"{p_u35*100:.1f}%", "Alt Odds": f"{1/p_u35:.2f}"},
                {"Market": "BTTS (Yes)", "Probability": f"{p_btts_yes*100:.1f}%", "Fair Odds": f"{1/p_btts_yes:.2f}", "Alternative": "BTTS (No)", "Alt Prob": f"{p_btts_no*100:.1f}%", "Alt Odds": f"{1/p_btts_no:.2f}"},
                {"Market": f"{h2h_home} Clean Sheet", "Probability": f"{p_cs_home*100:.1f}%", "Fair Odds": f"{1/max(0.001, p_cs_home):.2f}", "Alternative": f"{h2h_away} Clean Sheet", "Alt Prob": f"{p_cs_away*100:.1f}%", "Alt Odds": f"{1/max(0.001, p_cs_away):.2f}"},
                {"Market": "Double Chance (1X)", "Probability": f"{p_1x*100:.1f}%", "Fair Odds": f"{1/p_1x:.2f}", "Alternative": "Double Chance (X2)", "Alt Prob": f"{p_x2*100:.1f}%", "Alt Odds": f"{1/p_x2:.2f}"}
            ]
            st.dataframe(pd.DataFrame(mkt_data), use_container_width=True, height=270, hide_index=True)
            
            if match_pred is not None and 'Explanation' in match_pred and pd.notna(match_pred['Explanation']):
                st.info(f"💡 **Model Contributing Factors**:\n\n{match_pred['Explanation']}")
                
        # Historical H2H Meetings
        st.write("---")
        st.markdown(f"### 📜 Historical Head-to-Head Record ({h2h_home} vs. {h2h_away})")
        if hist_df is not None and not hist_df.empty:
            h2h_matches = hist_df[
                ((hist_df['HomeTeam'] == h2h_home) & (hist_df['AwayTeam'] == h2h_away)) |
                ((hist_df['HomeTeam'] == h2h_away) & (hist_df['AwayTeam'] == h2h_home))
            ].copy()
            
            if h2h_matches.empty:
                st.write(f"No previous Premier League meetings recorded between {h2h_home} and {h2h_away} in the historical database (2015-2026).")
            else:
                total_m = len(h2h_matches)
                h_wins = len(h2h_matches[((h2h_matches['HomeTeam'] == h2h_home) & (h2h_matches['FTR'] == 'H')) | ((h2h_matches['AwayTeam'] == h2h_home) & (h2h_matches['FTR'] == 'A'))])
                draws = len(h2h_matches[h2h_matches['FTR'] == 'D'])
                a_wins = len(h2h_matches[((h2h_matches['HomeTeam'] == h2h_away) & (h2h_matches['FTR'] == 'H')) | ((h2h_matches['AwayTeam'] == h2h_away) & (h2h_matches['FTR'] == 'A'))])
                
                col_h1, col_h2, col_h3, col_h4 = st.columns(4)
                with col_h1:
                    st.metric("Total Meetings (Since 2015)", total_m)
                with col_h2:
                    st.metric(f"{h2h_home} Wins", f"{h_wins} ({h_wins/total_m*100:.0f}%)")
                with col_h3:
                    st.metric("Draws", f"{draws} ({draws/total_m*100:.0f}%)")
                with col_h4:
                    st.metric(f"{h2h_away} Wins", f"{a_wins} ({a_wins/total_m*100:.0f}%)")
                    
                # Recent encounters table
                h2h_matches['Date'] = pd.to_datetime(h2h_matches['Date'], format='mixed', dayfirst=True)
                recent_h2h = h2h_matches.sort_values('Date', ascending=False).head(8)
                recent_display = []
                for _, r in recent_h2h.iterrows():
                    recent_display.append({
                        'Season': r['Season'],
                        'Date': r['Date'].strftime('%d %b %Y'),
                        'Home': r['HomeTeam'],
                        'Score': f"{int(r['FTHG'])} - {int(r['FTAG'])}",
                        'Away': r['AwayTeam'],
                        'Outcome': f"{r['HomeTeam']} Win" if r['FTR'] == 'H' else (f"{r['AwayTeam']} Win" if r['FTR'] == 'A' else 'Draw')
                    })
                st.dataframe(pd.DataFrame(recent_display), use_container_width=True, hide_index=True)
        else:
            st.info("Historical match database not loaded.")


# ------------------------------------------------------------------------------
# TAB TRANSFERS: SQUAD TRANSFERS & SIGNINGS
# ------------------------------------------------------------------------------
with tab_transfers:
    st.subheader("🔄 Premier League Transfer Activity & Squad Updates")
    st.write("Explore all new player arrivals (signings) and departures across the 20 Premier League clubs.")
    
    if transfers_df is None or transfers_df.empty:
        st.info("No transfers data found in data/raw/transfers.csv.")
    else:
        teams_list = sorted(list(transfers_df['Team'].unique()))
        selected_transfer_team = st.selectbox("Select EPL Club to View Squad Activity:", teams_list)
        
        team_transfers = transfers_df[transfers_df['Team'] == selected_transfer_team]
        ins = team_transfers[team_transfers['TransferType'] == 'In']
        outs = team_transfers[team_transfers['TransferType'] == 'Out']
        
        total_in_fee = ins['Fee_M_Euros'].sum()
        total_out_fee = outs['Fee_M_Euros'].sum()
        net_spend = total_in_fee - total_out_fee
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric(label="📥 New Signings", value=len(ins))
        with col_m2:
            st.metric(label="📤 Players Departed", value=len(outs))
        with col_m3:
            st.metric(label="💰 Gross Spend", value=f"€{total_in_fee:.1f}M")
        with col_m4:
            st.metric(label="📊 Net Spend", value=f"€{net_spend:+.1f}M")
            
        st.write("---")
        col_in_list, col_out_list = st.columns(2)
        
        with col_in_list:
            st.markdown(f"### 📥 {selected_transfer_team} New Signings (Arrivals)")
            if ins.empty:
                st.write("No major incoming signings recorded.")
            else:
                for _, row in ins.iterrows():
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 14px; margin-bottom: 10px;">
                        <span style="font-weight: 700; font-size: 1.1rem; color: #10b981;">⚽ {row['Player']}</span> ({row['Position']})<br/>
                        <span style="color: #94a3b8; font-size: 0.9rem;">Joined from: <strong>{row['OtherClub']}</strong> | Fee: <strong>€{row['Fee_M_Euros']:.1f}M</strong></span>
                    </div>
                    """, unsafe_allow_html=True)
                    
        with col_out_list:
            st.markdown(f"### 📤 {selected_transfer_team} Departures (Players Left)")
            if outs.empty:
                st.write("No major departures recorded.")
            else:
                for _, row in outs.iterrows():
                    st.markdown(f"""
                    <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 14px; margin-bottom: 10px;">
                        <span style="font-weight: 700; font-size: 1.1rem; color: #ef4444;">🚪 {row['Player']}</span> ({row['Position']})<br/>
                        <span style="color: #94a3b8; font-size: 0.9rem;">Joined: <strong>{row['OtherClub']}</strong> | Fee Received: <strong>€{row['Fee_M_Euros']:.1f}M</strong></span>
                    </div>
                    """, unsafe_allow_html=True)
                    
        st.write("---")
        st.markdown("### 🌟 League-Wide Marquee Transfers")
        top_signings = transfers_df[transfers_df['TransferType'] == 'In'].sort_values('Fee_M_Euros', ascending=False).head(10)
        st.dataframe(
            top_signings[['Team', 'Player', 'Position', 'OtherClub', 'Fee_M_Euros']].rename(columns={
                'OtherClub': 'Former Club',
                'Fee_M_Euros': 'Fee (€M)'
            }),
            use_container_width=True,
            hide_index=True
        )


# ------------------------------------------------------------------------------
# TAB 4: NARRATIVE INSIGHTS
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("EPL 2026/27 Simulated Season Storylines")
    
    # 1. Title Race Narrative
    st.markdown("### 🏆 The Title Race")
    top_3_teams = standings_df.head(3)
    text_title_race = f"The Monte Carlo simulation predicts a highly contested battle at the top of the league. "
    for r_idx, row in top_3_teams.iterrows():
        text_title_race += f"**{row['Team']}** enters the campaign as a major contender with a **{row['TitleWin%']:.1f}%** chance of winning the title (expected points of {int(row['ExpectedPoints'])}). "
    st.write(text_title_race)
    
    # 2. Relegation Battle Narrative
    st.markdown("### Relegation Danger")
    bottom_3_teams = standings_df.tail(3)
    text_relegation = f"At the bottom of the table, survival will be a fierce scrap. The bottom 3 teams with the highest relegation probability are: "
    for r_idx, row in bottom_3_teams.iterrows():
        text_relegation += f"**{row['Team']}** (**{row['Relegated%']:.1f}%** risk, expected rank of {int(row['ExpectedRank'])}), "
    text_relegation = text_relegation.rstrip(', ') + "."
    st.write(text_relegation)
    
    # 3. Surprise Packages & Financial Efficiency
    st.markdown("### 📈 Efficiency & Surprise Packages")
    st.write("We compare the expected final standings against squad market values to find teams performing above or below their financial weight.")
    
    if squad_df is not None:
        # Merge standings and squad values
        merged_squad = pd.merge(standings_df, squad_df, left_on='Team', right_on='Team')
        # Compute squad rank
        merged_squad['SquadValueRank'] = merged_squad['MarketValue_M_Euros'].rank(ascending=False).astype(int)
        merged_squad['Efficiency'] = merged_squad['SquadValueRank'] - merged_squad['ExpectedRank']
        
        # Outperformers (Positive Efficiency)
        outperformers = merged_squad[merged_squad['Efficiency'] > 1].sort_values('Efficiency', ascending=False)
        # Underperformers (Negative Efficiency)
        underperformers = merged_squad[merged_squad['Efficiency'] < -1].sort_values('Efficiency', ascending=True)
        
        col_out, col_under = st.columns(2)
        
        with col_out:
            st.success("💰 Smart Spending (Projected to Outperform Squad Value Rank)")
            out_list = []
            for _, row in outperformers.head(3).iterrows():
                out_list.append(f"**{row['Team']}**: Squad Rank {row['SquadValueRank']} vs. Projected Standings Rank {row['ExpectedRank']} (+{row['Efficiency']} positions)")
            if out_list:
                st.write("\n\n".join(out_list))
            else:
                st.write("No major outperformers projected.")
                
        with col_under:
            st.error("📉 Underachievers (Projected to Underperform Squad Value Rank)")
            under_list = []
            for _, row in underperformers.head(3).iterrows():
                under_list.append(f"**{row['Team']}**: Squad Rank {row['SquadValueRank']} vs. Projected Standings Rank {row['ExpectedRank']} ({row['Efficiency']} positions)")
            if under_list:
                st.write("\n\n".join(under_list))
            else:
                st.write("No major underachievers projected.")
    else:
        st.info("Upload/save squad market values to activate financial efficiency metrics.")

# Helper to load historical data and configs for Sandbox
@st.cache_resource
def load_sandbox_data():
    import pandas as pd
    import os
    from src.v2.config import get_v2_config
    
    matches_path = 'data/processed/matches_with_features.csv'
    fixtures_path = 'data/raw/fixtures_raw.csv'
    final_elos_path = 'data/processed/final_elos.csv'
    
    if os.path.exists(matches_path) and os.path.exists(fixtures_path) and os.path.exists(final_elos_path):
        df = pd.read_csv(matches_path)
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
        # Precompute congestion rest days
        from src.v2.congestion import add_congestion_features
        df = add_congestion_features(df)
        
        fixtures_df = pd.read_csv(fixtures_path)
        fixtures_df['Date'] = pd.to_datetime(fixtures_df['Date'])
        
        final_elos_df = pd.read_csv(final_elos_path)
        final_elos = dict(zip(final_elos_df['Team'], final_elos_df['Elo']))
        
        config = get_v2_config()
        
        return df, fixtures_df, final_elos, config
    return None, None, None, None

# ------------------------------------------------------------------------------
# TAB 5: WHAT-IF SANDBOX
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("🔮 Premier League What-If Sandbox")
    st.write("Simulate the entire season in real-time under custom scenarios. Adjust squad injuries or fire managers, and watch the title odds and expected points shift instantly.")
    
    # Load data
    df_hist, fixtures_df_hist, final_elos_map, v2_cfg = load_sandbox_data()
    
    if df_hist is None:
        st.warning("Historical data missing. Please verify features are built.")
    else:
        # Sandbox parameters
        col_s1, col_s2 = st.columns(2)
        active_teams = sorted(list(final_elos_map.keys()))
        
        with col_s1:
            st.markdown("### 🚑 Adjust Team Injuries")
            selected_injury_team = st.selectbox("Select Team to Injure", ["None"] + active_teams, key="sandbox_injury_team")
            
            injury_loss = 0.0
            if selected_injury_team != "None":
                injury_loss = st.slider(
                    "Injury Impact Factor (Deficit to squad availability)",
                    min_value=0.0,
                    max_value=0.4,
                    value=0.15,
                    step=0.05,
                    help="0.15 represents ~15% drop in team rating (e.g. key player like Haaland/Saka injured)"
                )
                
            st.markdown("### 👔 Manager Changes")
            selected_mgr_team = st.selectbox("Fire & Appoint New Manager for:", ["None"] + active_teams, key="sandbox_mgr_team")
            
        with col_s2:
            st.markdown("### 📊 Active Scenarios Summary")
            scenarios = []
            custom_injured = []
            custom_managers = {}
            
            if selected_injury_team != "None" and injury_loss > 0:
                scenarios.append(f"🚨 **{selected_injury_team}**: Key injury active (rating deficit: {int(injury_loss*100)}%)")
                custom_injured.append({'Team': selected_injury_team, 'Importance': injury_loss})
                
            if selected_mgr_team != "None":
                scenarios.append(f"👔 **{selected_mgr_team}**: Appointed a new manager today (triggers new manager bounce)")
                # Set appointment date to the start of the season so it qualifies as new manager
                custom_managers[selected_mgr_team] = '2026-08-01'
                
            if scenarios:
                for sc in scenarios:
                    st.markdown(sc)
            else:
                st.write("No active overlays. The simulator will match the pre-season default predictions.")
                
            num_sandbox_sims = st.number_input("Number of Simulations", min_value=100, max_value=2000, value=500, step=100)
            run_sim = st.button("🔮 Run Sandbox Simulation")
            
        if run_sim:
            with st.spinner("Running vectorized Monte Carlo season simulation..."):
                # Run simulations
                from src.pipeline_v2 import run_season_simulations_v2
                
                # Force enable features in config
                sandbox_config = v2_cfg.copy()
                if selected_injury_team != "None":
                    sandbox_config['injuries'] = True
                if selected_mgr_team != "None":
                    # Enable promoted/manager features
                    sandbox_config['promoted_handling'] = True
                    
                # Execute simulation
                res_df = run_season_simulations_v2(
                    fixtures_df=fixtures_df_hist.copy(),
                    final_elos=final_elos_map.copy(),
                    df_historical=df_hist.copy(),
                    config=sandbox_config,
                    num_sims=num_sandbox_sims,
                    dc_weight=0.5,
                    custom_injured_players=custom_injured,
                    custom_managers=custom_managers
                )
                
                # Merge with baseline to show differences
                comparison_df = pd.merge(
                    standings_df[['Team', 'ExpectedPoints', 'TitleWin%', 'Relegated%']],
                    res_df[['Team', 'ExpectedPoints', 'TitleWin%', 'Relegated%']],
                    on='Team',
                    suffixes=('_Baseline', '_WhatIf')
                )
                
                comparison_df['Points Shift'] = comparison_df['ExpectedPoints_WhatIf'] - comparison_df['ExpectedPoints_Baseline']
                comparison_df['Title Chance Shift'] = comparison_df['TitleWin%_WhatIf'] - comparison_df['TitleWin%_Baseline']
                
                # Format comparison table
                comparison_df = comparison_df.sort_values('ExpectedPoints_WhatIf', ascending=False).reset_index(drop=True)
                
                st.success("✅ Simulation completed! Compare findings below:")
                
                # Display styled differences
                st.dataframe(
                    comparison_df.style.format({
                        'ExpectedPoints_Baseline': '{:.1f}',
                        'ExpectedPoints_WhatIf': '{:.1f}',
                        'Points Shift': '{:+.1f}',
                        'TitleWin%_Baseline': '{:.1f}%',
                        'TitleWin%_WhatIf': '{:.1f}%',
                        'Title Chance Shift': '{:+.1f}%',
                        'Relegated%_Baseline': '{:.1f}%',
                        'Relegated%_WhatIf': '{:.1f}%'
                    }).background_gradient(subset=['Points Shift'], cmap='RdYlGn'),
                    use_container_width=True,
                    height=600
                )
