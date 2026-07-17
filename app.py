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

# Helper function to load simulated data
@st.cache_data
def load_data():
    preds_path = 'output/predictions.csv'
    standings_path = 'output/final_table_projection.csv'
    squad_path = 'data/raw/squad_values.csv'
    
    preds_df = pd.read_csv(preds_path) if os.path.exists(preds_path) else None
    standings_df = pd.read_csv(standings_path) if os.path.exists(standings_path) else None
    squad_df = pd.read_csv(squad_path) if os.path.exists(squad_path) else None
    
    return preds_df, standings_df, squad_df

preds_df, standings_df, squad_df = load_data()

# Check if data exists
if standings_df is None or preds_df is None:
    st.error("Error: Simulation outputs not found in `/output/`. Please run the pipeline script first using the terminal command: `python -m src.pipeline`")
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
        <div class="kpi-label">🏆 Title Favorite ({title_fav['TitleWin%']:.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-val">{top4_fav['Team']}</div>
        <div class="kpi-label">⭐ Top 4 Lock ({top4_fav['Top4%']:.1f}%)</div>
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Simulated Standings", 
    "📈 Standing Distributions", 
    "⚽ Match predictions", 
    "🔍 Narrative Insights",
    "🔮 What-If Sandbox"
])

# ------------------------------------------------------------------------------
# TAB 1: SIMULATED STANDINGS
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Projected Final League Table")
    st.write("This table shows the median final points, goal difference, and ranks across all 10,000 simulations, together with 90% confidence intervals.")
    
    # Styled dataframe
    # Helper to add colors
    def style_table(val):
        # We can format cells inside the st.dataframe styler
        pass
        
    # We display the standings_df with nice column names
    display_df = standings_df.copy()
    display_df.columns = [
        'Team', 'Exp Points', 'Points Range (90% CI)', 'Exp GD', 
        'GD Range', 'Exp Rank', 'Rank Range', 'Title Win %', 'Top 4 %', 'Relegation %'
    ]
    
    # Apply styling
    # Highlight zones: Title (Gold), Champions League (Blue), Relegation (Red)
    # We can style the dataframe using pandas Styler
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
    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    with col_f1:
        matchday_filter = st.selectbox(
            "Matchday", 
            ["All"] + sorted(list(preds_df['MatchDay'].unique()))
        )
    with col_f2:
        team_search = st.text_input("Filter by Team Name").strip()
        
    # Apply filters
    filtered_preds = preds_df.copy()
    if matchday_filter != "All":
        filtered_preds = filtered_preds[filtered_preds['MatchDay'] == matchday_filter]
        
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
            'HomeWin%': 'Home Win %',
            'Draw%': 'Draw %',
            'AwayWin%': 'Away Win %',
            'MostLikelyScore': 'Most Likely Score'
        })
        
        # Drop Explanation from dataframe columns to prevent cluttering the table
        cols_to_show = [c for c in display_preds.columns if c != 'Explanation']
        st.dataframe(
            display_preds[cols_to_show].style.format({
                'Home Win %': '{:.1f}%',
                'Draw %': '{:.1f}%',
                'Away Win %': '{:.1f}%'
            }), 
            use_container_width=True, 
            height=400,
            hide_index=True
        )
        
        # Interactive Match Detail card showing explainability breakdown
        st.write("---")
        st.subheader("🔍 Match Prediction Driver Inspector")
        st.write("Select any match from the filtered list below to view the human-readable explanation generated by the explainability module.")
        
        fixture_options = filtered_preds.index.tolist()
        if fixture_options:
            selected_idx = st.selectbox(
                "Select Match to Inspect:", 
                fixture_options,
                format_func=lambda i: f"GW {filtered_preds.loc[i, 'MatchDay']}: {filtered_preds.loc[i, 'HomeTeam']} vs. {filtered_preds.loc[i, 'AwayTeam']}"
            )
            
            match_row = filtered_preds.loc[selected_idx]
            
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
    st.markdown("### ⚠️ Relegation Danger")
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
