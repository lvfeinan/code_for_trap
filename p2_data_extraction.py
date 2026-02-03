"""
Modules for extracting environmental covariates and calculating ecological 
indicators within the two-tiered spatial grid system.
"""

import numpy as np
import pandas as pd
import pylandstats as pls
from tqdm.notebook import tqdm

# Import configuration and utilities from p1
from p1_config_utils import (
    path_data, path_nc, path_sample, 
    get_ar_circle, clip_sample, nearest_neighbor
)

class Calculate1YearLAMetric:
    """
    Calculates the Shannon Diversity Index for landscape patterns.
    """
    def __init__(self, year, radius=20):
        self.year = year
        self.radius = radius
        self.da_circle, self.ar_circle = get_ar_circle(radius)
        self.LA_result = []
        self.output_file = path_data / "sample_data/extract_data" / f"{self.year}_LA_metric.csv"
        
        if self.output_file.exists():
            self.df_LA_result = pd.read_csv(self.output_file, index_col=0)
        else:
            self.df_conflict = pd.read_csv(path_data / f"sample_Data/koppen_events_conflict_{year}.csv", index_col=0)
            self.df_non_conflict = pd.read_csv(path_data / f"sample_Data/koppen_events_non_conflict_{year}.csv", index_col=0)
            self._load_spatial_data()
            self._cal_LA()

    def _load_spatial_data(self):
        datasets = ["luc_lastyear", "luc_currentyear"]
        for _da in datasets:
            setattr(self, f"da_conflict_{_da}", clip_sample(path_nc / f"PSM25km/conflict_sample/{self.year}_{_da}.nc", self.da_circle))
            setattr(self, f"da_non_conflict_{_da}", clip_sample(path_nc / f"PSM25km/non_conflict_sample/{self.year}_{_da}.nc", self.da_circle))
    
    def _cal_LA(self):
        for _conflict in ["conflict", "non_conflict"]:
            for _year in ["currentyear", "lastyear"]:
                _da_luc = getattr(self, f"da_{_conflict}_luc_{_year}")
                for _idx in tqdm(list(_da_luc.idx.values), desc=f"Shannon {_conflict} {_year}"):
                    _ar_luc = _da_luc.sel(idx=_idx).drop_vars("idx").values
                    luc_types = np.unique(_ar_luc)
                    luc_types = luc_types[~np.isnan(luc_types)]
                    
                    if luc_types.shape[0] <= 1:
                        _shannon = 0
                    else:
                        _shannon = pls.Landscape(_ar_luc, res=(1000, 1000)).shannon_diversity_index()
                    
                    self.LA_result.append([_conflict, _year, _idx, _shannon])
                        
        self.df_LA_result = pd.DataFrame(self.LA_result, columns=["conflict", "year", "idx", "shannon"])\
            .pivot(index=["conflict", "idx"], columns="year", values="shannon")\
            .reset_index()
        self.df_LA_result.to_csv(self.output_file)


class Extract1YearData:
    """
    Extracts comprehensive zonal statistics (biophysical and socioeconomic) 
    for constructing the dataset for Propensity Score Matching (PSM).
    """
    def __init__(self, year, radius=20):
        self.year = year
        self.radius = radius
        self.da_circle, self.ar_circle = get_ar_circle(radius)
        self.output_file = path_data / "sample_data/extract_data" / f"{self.year}_zonal_combined.csv"
        
        self.df_conflict = pd.read_csv(path_data / f"sample_Data/koppen_events_conflict_{year}.csv", index_col=0)
        self.df_conflict_lastyear = pd.read_csv(path_data / f"sample_Data/koppen_events_conflict_{year-1}.csv", index_col=0)
        self.df_non_conflict = pd.read_csv(path_data / f"sample_Data/koppen_events_non_conflict_{year}.csv", index_col=0)

        self.df_conflict_nearest = nearest_neighbor(self.df_conflict, self.df_conflict_lastyear, return_dist=True)
        self.df_non_conflict_nearest = nearest_neighbor(self.df_non_conflict, self.df_conflict_lastyear, return_dist=True)
        
        if self.output_file.exists():
            self.df_zonal_combined = pd.read_csv(self.output_file, index_col=0)
        else:
            self._load_spatial_data()
            self._compute_zonal_stats()

        self._load_LA()
        self._seperate_zonal()
        
    def _load_spatial_data(self):
        datasets = ["luc_lastyear", "luc_currentyear", "road_dis", "boundary_dis", 
                    "npp_lastyear", "npp_currentyear", "pop", "pop_ly", "fire"]
        
        for _da in datasets:
            setattr(self, f"da_conflict_{_da}", clip_sample(path_nc / f"PSM25km/conflict_sample/{self.year}_{_da}.nc", self.da_circle))
            setattr(self, f"da_non_conflict_{_da}", clip_sample(path_nc / f"PSM25km/non_conflict_sample/{self.year}_{_da}.nc", self.da_circle))
            
    def _compute_zonal_stats(self):
        LAND_COVER_TYPES = {
            'crop': [12, 14], 'built': [13], 'forest': [1, 2, 3, 4, 5],
            'shrubland': [6, 7], 'grass': [10], 'wetland': [11], 'barren': [16],
        }

        def calculate_stats(df, da_prefix):
            stats = {
                'pop': getattr(self, f"da_{da_prefix}pop").sum(dim=["x", "y"]).values,
                'pop_ly': getattr(self, f"da_{da_prefix}pop_ly").sum(dim=["x", "y"]).values,
                'fire_area': getattr(self, f"da_{da_prefix}fire").sum(dim=["x", "y"]).values,
                'total_area': (~np.isnan(getattr(self, f"da_{da_prefix}luc_currentyear"))).sum(dim=["x", "y"]).values,
                'total_area_ly': (~np.isnan(getattr(self, f"da_{da_prefix}luc_lastyear"))).sum(dim=["x", "y"]).values,
                'dis2road': getattr(self, f"da_{da_prefix}road_dis").mean(dim=["x", "y"]).values,
                'dis2bound': getattr(self, f"da_{da_prefix}boundary_dis").mean(dim=["x", "y"]).values,
                'npp': getattr(self, f"da_{da_prefix}npp_currentyear").mean(dim=["x", "y"]).values,
                'npp_ly': getattr(self, f"da_{da_prefix}npp_lastyear").mean(dim=["x", "y"]).values,
                'dis2conflict': getattr(self, f"df_{da_prefix}nearest")["distance"].values,
            }

            current_year_da = getattr(self, f"da_{da_prefix}luc_currentyear")
            last_year_da = getattr(self, f"da_{da_prefix}luc_lastyear")

            for lc_type, codes in LAND_COVER_TYPES.items():
                stats[f'{lc_type}_area'] = current_year_da.isin(codes).sum(dim=["x", "y"]).values
                stats[f'{lc_type}_area_ly'] = last_year_da.isin(codes).sum(dim=["x", "y"]).values

            return df.copy().assign(**stats)

        def calculate_ratios(df):
            df = df.copy()
            for lc_type in LAND_COVER_TYPES:
                df[f'{lc_type}_ratio'] = df[f'{lc_type}_area'] / df.total_area * 100
                df[f'{lc_type}_ratio_ly'] = df[f'{lc_type}_area_ly'] / df.total_area_ly * 100
            return df

        def calculate_changes(df):
            df = df.copy()
            for lc_type in LAND_COVER_TYPES:
                df[f'{lc_type}_change'] = df[f'{lc_type}_ratio'] - df[f'{lc_type}_ratio_ly']
            return df

        self.df_conflict_zonal = self.df_conflict.pipe(calculate_stats, "conflict_").pipe(calculate_ratios).pipe(calculate_changes).assign(c=1)
        self.df_non_conflict_zonal = self.df_non_conflict.pipe(calculate_stats, "non_conflict_").pipe(calculate_ratios).pipe(calculate_changes).assign(c=0)
        
        self.df_zonal_combined = pd.concat([self.df_conflict_zonal, self.df_non_conflict_zonal], ignore_index=True)
        self.df_zonal_combined.to_csv(self.output_file)
    
    def _load_LA(self):
        self.df_LA = pd.read_csv(path_data / "sample_data/extract_data" / f"{self.year}_LA_metric.csv", index_col=0)\
            .assign(c=lambda x: x.conflict.map({"conflict": 1, "non_conflict": 0}))\
            .drop(columns=["conflict"])\
            .rename(columns={"currentyear": "shannon", "lastyear":"shannon_ly"})\
            .assign(shannon_change=lambda x: x.shannon - x.shannon_ly)
        self.df_zonal_combined = self.df_zonal_combined.merge(self.df_LA, how="left", on=["idx", "c"])
    
    def _seperate_zonal(self):
        self.df_zonal_heat = self.df_zonal_combined.query("event in [1, 4, 5]")
        self.df_zonal_drought = self.df_zonal_combined.query("event in [2, 4]")
        self.df_zonal_wet = self.df_zonal_combined.query("event in [3, 5]")