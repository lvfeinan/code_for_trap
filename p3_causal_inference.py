"""
Implementation of the counterfactual inference framework (PSM) and 
statistical testing to identify Climate-Conflict Traps.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_rel

class PSMMatcher:
    """
    Implements Propensity Score Matching (PSM) using Logistic Regression.
    """
    def __init__(self, df, 
                 covariates=['pop', 'dis2conflict', 'crop_ratio_ly', 'built_ratio_ly', 'dis2road', 'dis2bound'], 
                 treatment_col='c', caliper_ratio=0.25, _gp=None):
        self.df = df.dropna(subset=covariates).reset_index(drop=True)
        self.covariates = covariates
        self.treatment_col = treatment_col
        self.caliper_ratio = caliper_ratio
        self.propensity_scores = None
        self.ps_sd = None
        self.matched_df = None
        self._gp = _gp
        self.skip_num = 0
        
    def compute_propensity_score(self):
        X = self.df[self.covariates]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(solver='liblinear', random_state=0)
        model.fit(X_scaled, self.df[self.treatment_col])
        self.propensity_scores = model.predict_proba(X_scaled)[:, 1]
        self.ps_sd = np.std(self.propensity_scores)
        self.df['ps'] = self.propensity_scores

    def match(self):
        if self.propensity_scores is None:
            self.compute_propensity_score()

        caliper = self.caliper_ratio * self.ps_sd

        df_treat = self.df[self.df[self.treatment_col] == 1].copy()
        df_ctrl = self.df[self.df[self.treatment_col] == 0].copy()

        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(df_ctrl['ps'].values.reshape(-1, 1))

        matched_treat_idx = []
        matched_ctrl_idx = []

        for idx in df_treat.index:
            ps_t = self.df.at[idx, 'ps']
            dist, neighbors = nn.kneighbors([[ps_t]], return_distance=True)
            dist = dist[0][0]
            nearest_idx = neighbors[0][0]
            ctrl_i = df_ctrl.index[nearest_idx]

            if dist > caliper:
                self.skip_num += 1
                continue

            matched_treat_idx.append(idx)
            matched_ctrl_idx.append(ctrl_i)

        df_treat_matched = self.df.loc[matched_treat_idx].reset_index(drop=True)
        df_ctrl_matched = self.df.loc[matched_ctrl_idx].reset_index(drop=True)

        self.matched_df = df_treat_matched.join(df_ctrl_matched, rsuffix="_n")
        return self.matched_df

    def get_matched_data(self):
        return self.matched_df


def ttest_for_event(df_event, gp_col="koppen", min_size=50):
    """
    Conducts PSM and paired t-tests for a specific climate event type.
    """
    gp_for_event = df_event.query("c == 1").groupby(gp_col, as_index=False).size()\
        .query("size > @min_size")[gp_col].values
    
    df_event = df_event.query(f"{gp_col} in @gp_for_event")
    
    ttest_results = []
    
    for _gp in gp_for_event:
        df_event_gp = df_event.query(f"{gp_col} == @_gp")
        col_logistic = ['pop', 'dis2conflict', 'crop_ratio_ly', 'built_ratio_ly', 'dis2road', 'dis2bound']
        df_event_gp = df_event_gp.dropna(subset=col_logistic)     
        
        try:
            matcher = PSMMatcher(df_event_gp, _gp=_gp)
            _df_matched = matcher.match()
        except:
            print(f"Skipping group {_gp} due to insufficient data for matching.")
            continue
        
        luc_lst = ["fire_area", "forest_change",  "wetland_change", "shannon_change"]
        for luc_ in luc_lst:
            t_stat, t_p = ttest_rel(_df_matched[f"{luc_}"].values, _df_matched[f"{luc_}_n"].values)
            ttest_results.append([_gp, luc_, t_stat, t_p])
            
    return pd.DataFrame(ttest_results, columns=[gp_col, "luc", "t_stat", "t_p"])




# ==========================================
# Main Execution Entry Point
# ==========================================
# This block demonstrates how to run the full analysis pipeline using the classes defined above.

if __name__ == "__main__":
    from p2_data_extraction import Extract1YearData
    from p1_config_utils import dic_region, path_data
    
    # 1. Extraction & Calculation
    print("Step 1: Extracting Data and Calculating Metrics (2002-2023)...")
    data_lst = []
    for year in range(2002, 2024):
        # This step performs the spatial grid alignment and Shannon index calculation
        data1year = Extract1YearData(year, radius=20)
        data_lst.append(data1year)
    
    # 2. Aggregation
    print("Step 2: Aggregating Data by Event Type...")
    # Example: Aggregating Drought Events
    data_drought = pd.concat([d.df_zonal_drought for d in data_lst], ignore_index=True)\
        .assign(regi_short=lambda _df: _df["regi_pnas"].map(dic_region))
    
    # 3. Causal Inference (PSM + T-test)
    print("Step 3: Running Causal Inference (PSM + Paired T-tests)...")
    ttest_results = ttest_for_event(data_drought, gp_col="name_long", min_size=50)
    
    # 4. Output Results
    print("\nAnalysis Complete. Top significant results:")
    ttest_results.to_csv(path_data / "final_ttest_results_drought.csv")