import os
import pandas as pd
import numpy as np

class InjuryTracker:
    def __init__(self, csv_path='data/raw/injuries_mock.csv'):
        self.csv_path = csv_path
        self.injuries_df = None
        if os.path.exists(csv_path):
            try:
                self.injuries_df = pd.read_csv(csv_path)
                self.injuries_df['StartDate'] = pd.to_datetime(self.injuries_df['StartDate'])
                self.injuries_df['EndDate'] = pd.to_datetime(self.injuries_df['EndDate'])
            except Exception as e:
                print(f"Error loading injury database: {e}")

    def get_availability(self, team, date, custom_injured_players=None):
        """
        Compute availability score for a team on a given date.
        Availability = 1.0 - sum(Importance of active injuries)
        Capped at [0.6, 1.0]
        """
        loss = 0.0
        
        # Compute active injuries from historical mock database
        if self.injuries_df is not None and not self.injuries_df.empty:
            dt = pd.to_datetime(date)
            active = self.injuries_df[
                (self.injuries_df['Team'] == team) & 
                (self.injuries_df['StartDate'] <= dt) & 
                (self.injuries_df['EndDate'] >= dt)
            ]
            loss += active['Importance'].sum()
            
        # Append manual custom injured players from What-If dashboard overlays
        if custom_injured_players:
            for player in custom_injured_players:
                if player.get('Team') == team:
                    loss += player.get('Importance', 0.10)
                    
        avail = 1.0 - loss
        return max(0.6, min(1.0, avail))

    def adjust_expected_goals(self, lmbda, mu, home_avail, away_avail):
        """
        Adjust expected goals for a match:
        - Reduce lambda (Home Expected Goals) if HomeAvailability is low, increase it if AwayAvailability is low.
        - Reduce mu (Away Expected Goals) if AwayAvailability is low, increase it if HomeAvailability is low.
        """
        # Home goals adjust: home attack * away defense decay
        # Lower availability decreases attack power and defensive power
        # If home_avail is 0.85, lambda reduces. If away_avail is 0.85, lambda increases (defense is weaker)
        lmbda_adj = lmbda * (home_avail / away_avail)
        mu_adj = mu * (away_avail / home_avail)
        return max(0.01, lmbda_adj), max(0.01, mu_adj)
