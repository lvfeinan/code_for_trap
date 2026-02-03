"""
Configuration settings, directory paths, and utility functions for spatial analysis.
This module handles the environment setup and basic geometric operations used
for grid alignment and spatial sampling.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from sklearn.neighbors import BallTree
from shapely.geometry import box

# ==========================================
# Path Configuration & Global Data Loading
# ==========================================
# Note: Paths are configured for the local environment. 
path_data = Path("E:/GEODATA/climate_war_trap")
path_sample = Path("E:/GEODATA/HANPP_data/global/PSM/sample_point")
path_nc = Path("E:/GEODATA/HANPP_data/global")
path_continent = Path("E:/GEODATA/HANPP_data/global/vector/globalmap_revise")
path_koppen = path_data / "koppen_geiger_tif"

# Load global datasets (Macro-scale grid)
# These are loaded here to be accessible across modules
try:
    da_koppen = xr.open_dataarray(path_koppen / "1991_2020/koppen_geiger_0p5.tif").sel(band=1).drop_vars("band")
    da_multi_event_m = xr.open_dataarray(path_data / "climate/multi_events_2000_2023.nc")
except Exception as e:
    print(f"Warning: Could not load global rasters. Check paths in p1_config_utils.py. Error: {e}")
    da_koppen = None
    da_multi_event_m = None

# Region mapping dictionary
dic_region = {
    "Northern Africa and Western Asia": "N. Africa W. Asia",
    "Sub-Saharan Africa": "Sub-Saharan Africa",
    "Central Asia and Russian Federation": "C. Asia",
    "Eastern Asia": "E. Asia",
    "Southern Asia": "S. Asia",
    "Southeastern Asia": "S. Asia",
    "Northern America": "N. America",
    "Latin America and the Caribbean": "Latin America",
    "Western Europe": "W. Europe",
    "Eastern and South-Eastern Europe": "E. Europe",
    "Oceania and Australia": "Oceania",
}

# Load World Map
try:
    gdf_world = gpd.read_file(path_continent / "map.shp").clip(box(-180, -60, 180, 85))
    gdf_world["regi_short"] = gdf_world["regi_pnas"].map(dic_region)
except Exception:
    gdf_world = None

# ==========================================
# Utility Functions
# ==========================================

def get_ar_circle(radius):
    """
    Generate a circular mask for the Micro-scale grid (~1km) sampling.
    """
    radius_clip = {
        3: 2.54, 4: 3.53, 5: 4.49, 6: 5.52, 7: 6.52, 8: 7.49, 9: 8.52, 
        10: 9.49, 11: 10.52, 12: 11.49, 13: 12.49, 14: 13.51, 15: 14.49, 
        16: 15.52, 17: 16.46, 18: 17.49, 19: 18.49, 20: 19.47, 
        21: 20.49, 22: 21.49, 23: 22.49, 24: 23.49, 25: 24.43
    }[radius]
    
    gdf_circle = gpd.GeoDataFrame(
        {}, geometry=gpd.GeoDataFrame({}, geometry=gpd.points_from_xy([0], [0])).buffer(radius_clip), crs="epsg:4326"
    )
    
    da_circle = xr.DataArray(
        np.ones((radius * 2 + 1, radius * 2 + 1)), 
        coords={"y": np.arange(-radius, radius + 1), "x": np.arange(-radius, radius + 1)}
    ).rio.write_crs("epsg:4326").rio.clip(gdf_circle.geometry, all_touched=True, drop=False).fillna(0).astype(np.uint8)
    
    return da_circle, da_circle.values

def clip_sample(nc_file, da_circle):
    """
    Clip the spatial data using the generated circular mask.
    """
    da_ = xr.open_dataarray(nc_file)
    return xr.where(da_circle == 1, da_.sel(x=da_circle.x, y=da_circle.y), np.nan)

def get_nearest(src_points, candidates, k_neighbors=1):
    """
    Find nearest neighbors using BallTree (Haversine metric).
    """
    tree = BallTree(candidates, leaf_size=15, metric='haversine')
    distances, indices = tree.query(src_points, k=k_neighbors)
    distances = distances.transpose()
    indices = indices.transpose()
    closest = indices[0]
    closest_dist = distances[0]
    return (closest, closest_dist)

def nearest_neighbor(left_df, right_df, return_dist=False):
    """
    Calculate spatial lag variables (e.g., distance to previous conflict).
    """
    right = right_df.copy().reset_index(drop=True)
    left_radians = np.array(left_df.apply(lambda _df: (_df.x * np.pi / 180, _df.y * np.pi / 180), axis=1).to_list())
    right_radians = np.array(right_df.apply(lambda _df: (_df.x * np.pi / 180, _df.y * np.pi / 180), axis=1).to_list())
    
    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)
    closest_points = right.loc[closest].reset_index(drop=True)
    
    if return_dist:
        earth_radius = 6371000  # meters
        closest_points['distance'] = dist * earth_radius
        
    return closest_points