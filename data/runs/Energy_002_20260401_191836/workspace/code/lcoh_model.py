#!/usr/bin/env python3
"""
African Green Hydrogen to Europe: Geospatial Levelized Cost Model
=================================================================
Transparent geospatial LCOH model estimating the delivered cost of African
green hydrogen to Europe (via ammonia shipping and reconversion) by 2030
under multiple financing and policy scenarios.

Methodology based on:
- GeoH2 model (Halloran et al., 2024, MethodsX)
- Kenya geospatial LCOH model (Müller et al., 2023, Applied Energy)
- Cost of capital for RE projects (Steffen, 2020, Energy Economics)
- Interest rate effects on RE costs (Schmidt et al., 2019, Nature Sustainability)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import geopandas as gpd
from shapely.geometry import Point
import warnings
import os

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Energy_002_20260401_191836'
DATA_DIR   = f'{WORKSPACE}/data'
OUTPUT_DIR = f'{WORKSPACE}/outputs'
IMAGES_DIR = f'{WORKSPACE}/report/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# TECHNO-ECONOMIC PARAMETERS  (2030 projections)
# ─────────────────────────────────────────────────────────────
P = {
    # ── Solar PV ──────────────────────────────────────────────
    # Based on GeoH2 appendix (2024 values) adjusted for 2030 cost reduction
    # IRENA projects ~40% CAPEX reduction 2022→2030 for solar PV in Africa
    'solar_capex'       : 750_000,   # €/MW  (750 €/kW,  2030)
    'solar_opex_frac'   : 0.015,     # 1.5 % of CAPEX/yr
    'solar_lifetime'    : 25,        # years

    # Scaling of normalised potential (0–1) → annual full-load hours
    # Southern Africa: global horizontal irradiance ~2 000–2 500 kWh/m²/yr
    # CF range ≈ 0.22–0.28  →  FLH ≈ 1 900–2 450 h/yr
    'solar_flh_max'     : 2400,     # h/yr  (FLH when theo_pv = 1.0)

    # ── Onshore Wind ──────────────────────────────────────────
    'wind_capex'        : 1_100_000, # €/MW  (1 100 €/kW, 2030)
    'wind_opex_frac'    : 0.020,
    'wind_lifetime'     : 25,
    'wind_flh_max'      : 5_000,     # h/yr  (coastal Namibia can reach ~0.55 CF)

    # ── Electrolyser (PEM, 2030) ──────────────────────────────
    # From Kenya paper (2030): CAPEX ≤ 600 €/kW; efficiency 70 % LHV
    'ely_capex'         : 600_000,   # €/MW_el input
    'ely_opex_frac'     : 0.020,
    'ely_lifetime'      : 20,
    'ely_efficiency'    : 0.70,      # MWh_H₂_LHV / MWh_el
    'h2_lhv_kwh'        : 33.3,      # kWh/kg_H₂ (LHV)

    # ── Compressed H₂ Storage (buffer, 500 bar) ───────────────
    'storage_capex_per_mwh' : 21_700, # €/MWh_H₂  (GeoH2 appendix)
    'storage_days'      : 2,          # days of buffer
    'storage_lifetime'  : 20,

    # ── Water (desalination or freshwater treatment) ───────────
    'water_demand'      : 21,        # L/kg_H₂
    'water_freshwater_elec': 0.4,    # kWh/m³
    'water_ocean_elec'  : 3.7,       # kWh/m³ (desalination)
    'water_unit_cost'   : 1.25,      # €/m³
    'water_transport'   : 0.10,      # €/100 km/m³

    # ── Road infrastructure ────────────────────────────────────
    # GeoH2 appendix: short roads 626 k€/km, long roads 482 k€/km
    'road_capex_short'  : 626_478,   # €/km  (< 10 km)
    'road_capex_long'   : 481_867,   # €/km  (≥ 10 km)
    'road_opex'         : 7_150,     # €/km/yr
    'road_lifetime'     : 50,

    # ── Ammonia synthesis (at production site/port) ────────────
    # IRENA (2020) "Green Hydrogen Cost Reduction": HB synthesis ~450 €/kW_H₂_input
    # (includes air separation, synthesis loop, refrigeration, storage)
    # Electricity demand: 2.8 kWh_el/kg_H₂  (GeoH2 appendix, Halloran et al. 2024)
    'nh3_synth_elec'    : 2.8,       # kWh_el/kg_H₂
    'nh3_synth_capex'   : 450_000,   # €/MW_H₂_input (450 €/kW — IRENA 2030 projection)
    'nh3_synth_opex_frac': 0.020,
    'nh3_synth_lifetime': 25,

    # ── Truck transport of NH₃ to port ────────────────────────
    # GeoH2 appendix: NH₃ trailer 2 600 kg_H₂, 210 k€, 1.5 h load
    'truck_speed'       : 70,        # km/h
    'nh3_trailer_cap'   : 2_600,     # kg_H₂_eq per trailer
    'truck_fuel_l100km' : 35,        # L/100 km
    'diesel_price'      : 1.50,      # €/L
    'driver_wage'       : 2.85,      # €/h
    'truck_capex'       : 160_000,   # €
    'trailer_capex'     : 210_000,   # €
    'truck_lifetime'    : 8,
    'trailer_lifetime'  : 12,
    'truck_opex_frac'   : 0.12,
    'trailer_opex_frac' : 0.02,
    'loading_time_h'    : 1.5,       # h (one-way loading + unloading)

    # ── Shipping  (Africa → Rotterdam, ~12 000 km) ─────────────
    # Based on Müller et al. (2023): Mombasa→Rotterdam NH₃ ≈ 0.39 €/kg_H₂
    # Slightly scaled up for West/South African ports (similar distance)
    'nh3_shipping'      : 0.40,      # €/kg_H₂
    'port_handling'     : 0.10,      # €/kg_H₂

    # ── NH₃ reconversion at Rotterdam ─────────────────────────
    # Müller et al. (2023): ammonia cracking ≈ 1.17 €/kg_H₂ (2023)
    # Scaled to 2030 with technology learning: ~0.85 €/kg_H₂
    'reconversion_cost' : 0.85,      # €/kg_H₂

    # ── European offshore wind baseline (for comparison) ───────
    # 2030 North Sea: CAPEX ≈ 2 200 €/kW, CF ≈ 0.50
    'eu_offshore_capex' : 2_200_000, # €/MW
    'eu_offshore_opex_frac': 0.025,
    'eu_offshore_lifetime': 25,
    'eu_offshore_cf'    : 0.50,

    # ── Grey hydrogen reference (SMR) ─────────────────────────
    'smr_efficiency'    : 0.76,      # (LHV_H₂/LHV_NG)
    'smr_capex_per_kg'  : 0.30,      # €/kg_H₂ annualised CAPEX
    'smr_co2_intensity' : 9.0,       # kg_CO₂/kg_H₂ (unabated)
    'nat_gas_price_2030': 40.0,      # €/MWh (2030 estimate)
    'co2_price_baseline': 80.0,      # €/tCO₂ (EU ETS 2030)
    'co2_price_high'    : 150.0,     # €/tCO₂ (high policy scenario)
}

# ─────────────────────────────────────────────────────────────
# FINANCING SCENARIOS
# ─────────────────────────────────────────────────────────────
SCENARIOS = {
    'S1_Baseline': {
        'wacc_africa'  : 0.08,
        'wacc_eu'      : 0.05,
        'capex_mult'   : 1.00,
        'ship_mult'    : 1.00,
        'reconv_mult'  : 1.00,
        'label'        : 'Baseline\n(Africa 8%, EU 5%)',
        'short_label'  : 'Baseline',
        'color'        : '#2196F3',
        'linestyle'    : '-',
    },
    'S2_High_Finance': {
        'wacc_africa'  : 0.12,
        'wacc_eu'      : 0.07,
        'capex_mult'   : 1.00,
        'ship_mult'    : 1.00,
        'reconv_mult'  : 1.00,
        'label'        : 'High Finance Risk\n(Africa 12%, EU 7%)',
        'short_label'  : 'High Finance Risk',
        'color'        : '#F44336',
        'linestyle'    : '--',
    },
    'S3_De_Risked': {
        'wacc_africa'  : 0.05,
        'wacc_eu'      : 0.05,
        'capex_mult'   : 1.00,
        'ship_mult'    : 1.00,
        'reconv_mult'  : 1.00,
        'label'        : 'De-Risked Africa\n(Africa 5%, EU 5%)',
        'short_label'  : 'De-Risked',
        'color'        : '#4CAF50',
        'linestyle'    : '-.',
    },
    'S4_Optimistic': {
        'wacc_africa'  : 0.07,
        'wacc_eu'      : 0.05,
        'capex_mult'   : 0.80,   # -20% CAPEX (aggressive 2030 learning)
        'ship_mult'    : 0.90,
        'reconv_mult'  : 0.80,
        'label'        : 'Optimistic 2030\n(7%, −20% CAPEX)',
        'short_label'  : 'Optimistic 2030',
        'color'        : '#FF9800',
        'linestyle'    : ':',
    },
}

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def crf(wacc: float, n: int) -> float:
    """Capital Recovery Factor  CRF = r*(1+r)^n / ((1+r)^n - 1)"""
    if wacc == 0:
        return 1.0 / n
    return wacc * (1 + wacc)**n / ((1 + wacc)**n - 1)


def lcoe_pv(params: dict, wacc: float, capex_mult: float = 1.0) -> pd.Series:
    """
    Levelised Cost of Electricity for solar PV  [€/MWh]
    Uses theo_pv (0–1) → FLH_pv = theo_pv × solar_flh_max.
    Returns a pd.Series indexed to the input DataFrame.
    """
    capex = params['solar_capex'] * capex_mult    # €/MW
    opex  = capex * params['solar_opex_frac']     # €/MW/yr
    n     = params['solar_lifetime']
    flh   = df['theo_pv'] * params['solar_flh_max']   # h/yr
    annual_gen = flh                               # MWh/MW/yr
    return (capex * crf(wacc, n) + opex) / annual_gen   # €/MWh


def lcoe_wind(params: dict, wacc: float, capex_mult: float = 1.0) -> pd.Series:
    """Levelised Cost of Electricity for onshore wind  [€/MWh]"""
    capex = params['wind_capex'] * capex_mult
    opex  = capex * params['wind_opex_frac']
    n     = params['wind_lifetime']
    flh   = df['theo_wind'] * params['wind_flh_max']
    annual_gen = flh
    return (capex * crf(wacc, n) + opex) / annual_gen


def electrolyser_cost_per_kg(params: dict, wacc: float,
                              flh: pd.Series, capex_mult: float = 1.0) -> pd.Series:
    """
    Annualised electrolyser capital + opex cost  [€/kg_H₂]
    Parameters
    ----------
    flh  : annual full-load hours of the electrolyser [h/yr]
           assumed equal to the generating asset FLH (wind or solar)
    """
    capex  = params['ely_capex'] * capex_mult             # €/MW_el
    opex   = capex * params['ely_opex_frac']
    n      = params['ely_lifetime']
    eta    = params['ely_efficiency']                     # MWh_H₂/MWh_el
    lhv    = params['h2_lhv_kwh'] / 1000                 # MWh/kg_H₂
    # kg_H₂ produced per MW_el per hour at full load:
    kg_per_mw_hr = eta / lhv                              # kg/(MW_el·h)
    annual_kg = flh * kg_per_mw_hr                       # kg/MW_el/yr
    return (capex * crf(wacc, n) + opex) / annual_kg     # €/kg_H₂


def electricity_cost_per_kg(lcoe_elec: pd.Series, params: dict) -> pd.Series:
    """
    Electricity cost for electrolysis  [€/kg_H₂]
    = LCOE [€/MWh] × (LHV_H₂/eta_ely) [MWh_el/kg_H₂]
    """
    eta = params['ely_efficiency']
    lhv = params['h2_lhv_kwh'] / 1000   # MWh/kg_H₂
    mwh_el_per_kg = lhv / eta            # MWh_el/kg_H₂
    return lcoe_elec * mwh_el_per_kg     # €/kg_H₂


def h2_storage_cost_per_kg(params: dict, wacc: float,
                            flh: pd.Series, capex_mult: float = 1.0) -> pd.Series:
    """
    Buffer H₂ storage cost at production site  [€/kg_H₂]
    Sized for `storage_days` days of output at full capacity.
    """
    capex_per_mwh = params['storage_capex_per_mwh'] * capex_mult
    n   = params['storage_lifetime']
    lhv = params['h2_lhv_kwh'] / 1000   # MWh/kg_H₂
    days_per_year = 365
    eta = params['ely_efficiency']
    kg_per_mw_hr  = eta / lhv
    # Annual H₂ production per MW_el:
    annual_kg = flh * kg_per_mw_hr       # kg/yr per MW_el capacity
    # Storage capacity (per MW_el):  storage_days × (annual_kg / days_per_year)
    storage_mwh = (params['storage_days'] / days_per_year) * annual_kg * lhv   # MWh_H₂/MW_el
    storage_capex = capex_per_mwh * storage_mwh                # €/MW_el
    return (storage_capex * crf(wacc, n)) / annual_kg          # €/kg_H₂


def water_cost_per_kg(params: dict, lcoe_elec: pd.Series) -> pd.Series:
    """
    Water treatment / desalination cost  [€/kg_H₂]
    Uses minimum of freshwater (distance-weighted) and ocean desalination.
    """
    demand_m3 = params['water_demand'] / 1000.0      # m³/kg_H₂
    # Freshwater option
    fw_elec_cost = params['water_freshwater_elec'] * (lcoe_elec / 1000)  # €/m³ from own RE
    fw_transport = params['water_transport'] * (df['waterbody_dist_km'] / 100)  # €/m³
    fw_total = (fw_elec_cost + fw_transport + params['water_unit_cost']) * demand_m3

    # Ocean desalination option
    oc_elec_cost = params['water_ocean_elec'] * (lcoe_elec / 1000)
    oc_transport = params['water_transport'] * (df['ocean_dist_km'] / 100)
    oc_total = (oc_elec_cost + oc_transport + params['water_unit_cost']) * demand_m3

    return pd.concat([fw_total, oc_total], axis=1).min(axis=1)


def road_infra_cost_per_kg(params: dict, wacc: float,
                            annual_h2_t: float = 50_000) -> pd.Series:
    """
    Road construction cost amortised per kg_H₂  [€/kg_H₂]
    Only constructed where road_dist_km > 0.
    annual_h2_t : annual H₂ throughput [tonnes] used to amortise infrastructure.
    """
    n    = params['road_lifetime']
    opex = params['road_opex']
    # Capex per km
    capex_km = np.where(df['road_dist_km'] <= 10,
                        params['road_capex_short'],
                        params['road_capex_long'])
    total_capex = capex_km * df['road_dist_km']
    annual_cost = total_capex * crf(wacc, n) + opex * df['road_dist_km']
    annual_h2_kg = annual_h2_t * 1000
    return pd.Series(annual_cost / annual_h2_kg, index=df.index)


def nh3_synthesis_cost_per_kg(params: dict, wacc: float,
                               lcoe_elec: pd.Series, flh: pd.Series,
                               capex_mult: float = 1.0) -> pd.Series:
    """
    NH₃ synthesis cost  [€/kg_H₂]
    = Electricity cost + annualised CAPEX/OPEX

    Model: CAPEX is proportional to hydrogen input capacity (MW_H₂_input).
    IRENA (2020): ~450 €/kW_H₂_input for HB synthesis at 2030 scale.

    Parameters
    ----------
    flh    : electrolyser/plant annual full-load hours [h/yr]
    """
    # Electricity for NH₃ synthesis (Haber-Bosch electrical energy)
    elec = params['nh3_synth_elec'] * (lcoe_elec / 1000)   # €/kg_H₂

    # CAPEX-based component (per MW_H₂_input per year → per kg_H₂)
    capex = params['nh3_synth_capex'] * capex_mult           # €/MW_H₂
    opex  = capex * params['nh3_synth_opex_frac']
    n     = params['nh3_synth_lifetime']
    eta   = params['ely_efficiency']
    lhv   = params['h2_lhv_kwh'] / 1000                     # MWh/kg_H₂
    # kg_H₂ processed per MW_H₂_input at rated capacity:
    kg_per_mw_hr = eta / lhv
    # NH₃ synthesis operates at same utilisation as electrolyser
    annual_kg = flh * kg_per_mw_hr                          # kg/(MW·yr)
    capex_component = (capex * crf(wacc, n) + opex) / annual_kg  # €/kg_H₂

    return elec + capex_component


def truck_transport_cost_per_kg(params: dict, wacc: float) -> pd.Series:
    """
    NH₃ truck transport cost from production site to nearest port  [€/kg_H₂]
    Based on GeoH2 trucking model (Halloran et al., 2024).
    """
    dist = df['ocean_dist_km']  # km to ocean/port (proxy for export port distance)
    cap  = params['nh3_trailer_cap']  # kg_H₂ per load

    # Fuel + driver (variable costs per round trip)
    trip_time_h    = dist / params['truck_speed'] + 2 * params['loading_time_h']
    fuel_cost      = (params['truck_fuel_l100km'] / 100) * dist * params['diesel_price']
    driver_cost    = trip_time_h * params['driver_wage']
    variable_rt    = fuel_cost + driver_cost  # €/round-trip per trailer
    variable_per_kg = variable_rt / cap       # €/kg_H₂

    # Capital costs (truck + trailer), amortised per kg_H₂
    # Assume all trucks run 24/7, 365 days  → trips per year
    round_trip_h   = 2 * trip_time_h
    annual_trips   = (365 * 24) / round_trip_h
    annual_kg_per_truck = annual_trips * cap
    truck_ann  = params['truck_capex']   * (crf(wacc, params['truck_lifetime'])
                                            + params['truck_opex_frac'])
    trail_ann  = params['trailer_capex'] * (crf(wacc, params['trailer_lifetime'])
                                            + params['trailer_opex_frac'])
    capital_per_kg = (truck_ann + trail_ann) / annual_kg_per_truck

    return variable_per_kg + capital_per_kg


def lcoh_delivered(params: dict, scenario: dict) -> dict:
    """
    Compute all cost components and total delivered LCOH for a given scenario.

    Returns a dict of pd.Series (one value per hexagon):
        'lcoe_selected'   : LCOE of cheapest electricity source  [€/MWh]
        'tech_selected'   : 'solar' or 'wind'
        'c_electricity'   : electricity cost component           [€/kg_H₂]
        'c_electrolyser'  : electrolyser CAPEX+OPEX              [€/kg_H₂]
        'c_storage'       : H₂ buffer storage                    [€/kg_H₂]
        'c_water'         : water treatment/desalination          [€/kg_H₂]
        'c_road_infra'    : road infrastructure                  [€/kg_H₂]
        'lcoh_production' : production LCOH (sum above)          [€/kg_H₂]
        'c_nh3_synth'     : NH₃ synthesis                        [€/kg_H₂]
        'c_road_transport': truck transport to port              [€/kg_H₂]
        'c_shipping'      : ocean shipping                       [€/kg_H₂]
        'c_port'          : port handling                        [€/kg_H₂]
        'c_reconversion'  : NH₃ reconversion at Rotterdam        [€/kg_H₂]
        'lcoh_delivered'  : total delivered LCOH                 [€/kg_H₂]
    """
    w_af  = scenario['wacc_africa']
    w_eu  = scenario['wacc_eu']
    cmult = scenario['capex_mult']
    smult = scenario['ship_mult']
    rmult = scenario['reconv_mult']

    # ── Electricity generation ────────────────────────────────
    le_pv   = lcoe_pv(params, w_af, cmult)      # €/MWh
    le_wind = lcoe_wind(params, w_af, cmult)    # €/MWh

    # Select cheapest electricity source per hexagon
    use_solar = le_pv <= le_wind
    lcoe_sel  = pd.Series(np.where(use_solar, le_pv, le_wind), index=df.index)
    tech_sel  = pd.Series(np.where(use_solar, 'solar', 'wind'),  index=df.index)

    # Electrolyser FLH follows generating asset
    flh_solar = df['theo_pv']   * params['solar_flh_max']
    flh_wind  = df['theo_wind'] * params['wind_flh_max']
    flh_ely   = pd.Series(np.where(use_solar, flh_solar, flh_wind), index=df.index)

    # ── Cost components at production site ───────────────────
    c_elec  = electricity_cost_per_kg(lcoe_sel, params)          # €/kg_H₂
    c_ely   = electrolyser_cost_per_kg(params, w_af, flh_ely, cmult)
    c_stor  = h2_storage_cost_per_kg(params, w_af, flh_ely, cmult)
    c_water = water_cost_per_kg(params, lcoe_sel)
    c_road  = road_infra_cost_per_kg(params, w_af)
    lcoh_prod = c_elec + c_ely + c_stor + c_water + c_road

    # ── Supply chain to Rotterdam ─────────────────────────────
    c_nh3   = nh3_synthesis_cost_per_kg(params, w_af, lcoe_sel, flh_ely, cmult)
    c_truck = truck_transport_cost_per_kg(params, w_af)
    c_ship  = params['nh3_shipping']  * smult
    c_port  = params['port_handling'] * smult
    c_recon = params['reconversion_cost'] * rmult

    lcoh_del = lcoh_prod + c_nh3 + c_truck + c_ship + c_port + c_recon

    return {
        'lcoe_pv'          : le_pv,
        'lcoe_wind'        : le_wind,
        'lcoe_selected'    : lcoe_sel,
        'tech_selected'    : tech_sel,
        'flh_ely'          : flh_ely,
        'c_electricity'    : c_elec,
        'c_electrolyser'   : c_ely,
        'c_storage'        : c_stor,
        'c_water'          : c_water,
        'c_road_infra'     : c_road,
        'lcoh_production'  : lcoh_prod,
        'c_nh3_synth'      : c_nh3,
        'c_road_transport' : c_truck,
        'c_shipping'       : pd.Series(c_ship,  index=df.index),
        'c_port'           : pd.Series(c_port,  index=df.index),
        'c_reconversion'   : pd.Series(c_recon, index=df.index),
        'lcoh_delivered'   : lcoh_del,
    }


def eu_green_h2_cost(params: dict, scenario: dict) -> dict:
    """
    European green H₂ production cost via offshore wind + electrolyser.
    Returns dict with LCOH_production and LCOH_at_demand.
    """
    w_eu  = scenario['wacc_eu']
    cmult = scenario['capex_mult']

    capex_w = params['eu_offshore_capex'] * cmult
    opex_w  = capex_w * params['eu_offshore_opex_frac']
    flh_w   = params['eu_offshore_cf'] * 8760  # h/yr
    le_w    = (capex_w * crf(w_eu, params['eu_offshore_lifetime']) + opex_w) / flh_w

    # Electrolyser (European WACC)
    flh_ely = pd.Series([flh_w] * len(df), index=df.index)
    c_ely   = electrolyser_cost_per_kg(params, w_eu, flh_ely, cmult).iloc[0]
    c_elec  = electricity_cost_per_kg(pd.Series([le_w]), params).iloc[0]

    # Simple water + miscellaneous
    c_misc  = 0.10  # €/kg_H₂

    lcoh_prod = c_elec + c_ely + c_misc
    # Storage + local distribution within Europe
    lcoh_at_demand = lcoh_prod + 0.30   # €/kg_H₂ local distribution

    return {
        'lcoe_offshore'  : le_w,
        'c_electricity'  : c_elec,
        'c_electrolyser' : c_ely,
        'lcoh_production': lcoh_prod,
        'lcoh_at_demand' : lcoh_at_demand,
    }


def grey_h2_cost_with_carbon(params: dict, co2_price: float) -> float:
    """
    Grey H₂ (SMR) cost including carbon price  [€/kg_H₂]
    """
    # Fuel cost component
    ng_kwh_per_kg = params['h2_lhv_kwh'] / params['smr_efficiency']  # kWh_NG/kg_H₂
    fuel_cost = ng_kwh_per_kg * params['nat_gas_price_2030'] / 1000  # €/kg_H₂
    carbon_cost = params['smr_co2_intensity'] * co2_price / 1000     # €/kg_H₂ (€/t × t/1000kg)
    return fuel_cost + params['smr_capex_per_kg'] + carbon_cost


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(f'{DATA_DIR}/hex_final_NA_min.csv')
df.set_index('hex_id', inplace=True)

print(f"  Hexagons: {len(df)}")
print(f"  Columns : {list(df.columns)}")
print(f"  Lat range: {df['lat'].min():.1f}°–{df['lat'].max():.1f}°N")
print(f"  Lon range: {df['lon'].min():.1f}°–{df['lon'].max():.1f}°E")
print(f"  theo_pv  : {df['theo_pv'].min():.3f}–{df['theo_pv'].max():.3f}")
print(f"  theo_wind: {df['theo_wind'].min():.3f}–{df['theo_wind'].max():.3f}")
print(f"  ocean_dist: {df['ocean_dist_km'].min():.0f}–{df['ocean_dist_km'].max():.0f} km")

# Load Africa shapefile
gdf_world = gpd.read_file(f'{DATA_DIR}/africa_map/ne_10m_admin_0_countries.shp')
gdf_africa = gdf_world[gdf_world['CONTINENT'] == 'Africa'].copy()

# Convert hexagons to GeoDataFrame
geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
gdf_hex   = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs='EPSG:4326')

# ─────────────────────────────────────────────────────────────
# RUN SCENARIOS
# ─────────────────────────────────────────────────────────────
print("\nRunning scenarios...")
results = {}
eu_results = {}
for sc_name, sc in SCENARIOS.items():
    print(f"  {sc_name}...")
    results[sc_name]    = lcoh_delivered(P, sc)
    eu_results[sc_name] = eu_green_h2_cost(P, sc)

# ─────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────
print("\nSaving outputs...")
for sc_name, res in results.items():
    out = df.copy()
    for key, val in res.items():
        if hasattr(val, '__len__') and len(val) == len(df):
            out[key] = val.values if hasattr(val, 'values') else val
    out.to_csv(f'{OUTPUT_DIR}/results_{sc_name}.csv')

# Summary table
summary_rows = []
for sc_name, res in results.items():
    ld = res['lcoh_delivered']
    eu = eu_results[sc_name]
    row = {
        'scenario'            : SCENARIOS[sc_name]['short_label'],
        'min_delivered_eur_kg': round(ld.min(), 2),
        'median_delivered'    : round(ld.median(), 2),
        'max_delivered'       : round(ld.max(), 2),
        'eu_lcoh_production'  : round(eu['lcoh_production'], 2),
        'eu_lcoh_at_demand'   : round(eu['lcoh_at_demand'], 2),
        'cost_advantage_best' : round(eu['lcoh_at_demand'] - ld.min(), 2),
        'wacc_africa'         : SCENARIOS[sc_name]['wacc_africa'],
    }
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f'{OUTPUT_DIR}/scenario_summary.csv', index=False)
print(summary_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family'     : 'DejaVu Sans',
    'font.size'       : 11,
    'axes.titlesize'  : 12,
    'axes.labelsize'  : 11,
    'figure.dpi'      : 150,
})

# ── Fig 1: Data Overview ──────────────────────────────────────
print("\nGenerating Figure 1: Data overview...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Input Data Overview – African Green Hydrogen Sites (Namibia Region)',
             fontsize=14, fontweight='bold', y=0.98)

vars_info = [
    ('theo_pv',            'Solar PV Potential (normalised)',   'YlOrRd'),
    ('theo_wind',          'Wind Potential (normalised)',        'Blues'),
    ('ocean_dist_km',      'Distance to Ocean (km)',            'RdPu'),
    ('road_dist_km',       'Distance to Road (km)',             'Greens'),
    ('grid_dist_km',       'Distance to Grid (km)',             'Oranges'),
    ('waterbody_dist_km',  'Distance to Water Body (km)',       'BuGn'),
]

for ax, (col, title, cmap) in zip(axes.flat, vars_info):
    gdf_africa.boundary.plot(ax=ax, linewidth=0.5, color='grey')
    gdf_africa.plot(ax=ax, color='#f5f5f0', edgecolor='grey', linewidth=0.3)
    sc = ax.scatter(df['lon'], df['lat'], c=df[col], cmap=cmap,
                    s=120, zorder=5, edgecolors='white', linewidths=0.3)
    plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(df['lon'].min()-3, df['lon'].max()+3)
    ax.set_ylim(df['lat'].min()-3, df['lat'].max()+3)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig01_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig01_data_overview.png")

# ── Fig 2: LCOE Maps (solar vs wind) ─────────────────────────
print("Generating Figure 2: LCOE comparison maps...")
sc_base = 'S1_Baseline'
res_base = results[sc_base]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Levelised Cost of Electricity (LCOE) by Technology – Baseline Scenario',
             fontsize=13, fontweight='bold')

data_plots = [
    (res_base['lcoe_pv'],    'Solar PV LCOE (€/MWh)',    'YlOrRd'),
    (res_base['lcoe_wind'],  'Wind LCOE (€/MWh)',         'Blues'),
    (res_base['lcoe_selected'], 'Selected (Cheapest) LCOE (€/MWh)', 'RdYlGn_r'),
]

for ax, (data, title, cmap) in zip(axes, data_plots):
    gdf_africa.plot(ax=ax, color='#f0f0eb', edgecolor='grey', linewidth=0.4)
    vmin, vmax = data.min(), data.max()
    sc = ax.scatter(df['lon'], df['lat'], c=data, cmap=cmap,
                    vmin=vmin, vmax=vmax, s=140, zorder=5,
                    edgecolors='white', linewidths=0.3)
    cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label('€/MWh', size=10)
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(df['lon'].min()-3, df['lon'].max()+3)
    ax.set_ylim(df['lat'].min()-3, df['lat'].max()+3)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig02_lcoe_maps.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig02_lcoe_maps.png")

# ── Fig 3: LCOH Production Maps (4 scenarios) ────────────────
print("Generating Figure 3: LCOH production maps...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Levelised Cost of Hydrogen (LCOH) at Production Site – Four Scenarios',
             fontsize=14, fontweight='bold')

vmin_all = min(res['lcoh_production'].min() for res in results.values())
vmax_all = max(res['lcoh_production'].max() for res in results.values())

for ax, (sc_name, sc) in zip(axes.flat, SCENARIOS.items()):
    res = results[sc_name]
    data = res['lcoh_production']
    gdf_africa.plot(ax=ax, color='#f0f0eb', edgecolor='grey', linewidth=0.4)
    sc_plot = ax.scatter(df['lon'], df['lat'], c=data, cmap='RdYlGn_r',
                         vmin=vmin_all, vmax=vmax_all, s=150, zorder=5,
                         edgecolors='white', linewidths=0.3)
    cb = plt.colorbar(sc_plot, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label('€/kg_H₂', size=10)
    ax.set_title(f"{sc['short_label']}\n(WACC_Africa = {sc['wacc_africa']*100:.0f}%)",
                 fontweight='bold')
    ax.set_xlim(df['lon'].min()-3, df['lon'].max()+3)
    ax.set_ylim(df['lat'].min()-3, df['lat'].max()+3)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.grid(True, alpha=0.3)

    # Annotate min/max
    idx_min = data.idxmin()
    idx_max = data.idxmax()
    ax.annotate(f"Min: {data.min():.2f}", xy=(df.loc[idx_min,'lon'], df.loc[idx_min,'lat']),
                xytext=(8, 8), textcoords='offset points', fontsize=9,
                color='darkgreen', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=0.8))
    ax.annotate(f"Max: {data.max():.2f}", xy=(df.loc[idx_max,'lon'], df.loc[idx_max,'lat']),
                xytext=(8, -14), textcoords='offset points', fontsize=9,
                color='darkred', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8))

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig03_lcoh_production_maps.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig03_lcoh_production_maps.png")

# ── Fig 4: Delivered Cost Maps ────────────────────────────────
print("Generating Figure 4: Delivered cost maps...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Total Delivered Cost of Green H₂ to Rotterdam – Four Scenarios',
             fontsize=14, fontweight='bold')

vmin_d = min(res['lcoh_delivered'].min() for res in results.values())
vmax_d = max(res['lcoh_delivered'].max() for res in results.values())

for ax, (sc_name, sc) in zip(axes.flat, SCENARIOS.items()):
    res = results[sc_name]
    data = res['lcoh_delivered']
    eu   = eu_results[sc_name]
    gdf_africa.plot(ax=ax, color='#f0f0eb', edgecolor='grey', linewidth=0.4)
    sc_plot = ax.scatter(df['lon'], df['lat'], c=data, cmap='RdYlGn_r',
                         vmin=vmin_d, vmax=vmax_d, s=150, zorder=5,
                         edgecolors='white', linewidths=0.3)
    cb = plt.colorbar(sc_plot, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label('€/kg_H₂', size=10)
    ax.set_title(f"{sc['short_label']}\n(WACC_Africa={sc['wacc_africa']*100:.0f}%, "
                 f"EU ref={eu['lcoh_at_demand']:.2f} €/kg)",
                 fontweight='bold')
    ax.set_xlim(df['lon'].min()-3, df['lon'].max()+3)
    ax.set_ylim(df['lat'].min()-3, df['lat'].max()+3)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig04_delivered_cost_maps.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig04_delivered_cost_maps.png")

# ── Fig 5: Cost Breakdown (waterfall) – Best & Worst Sites ───
print("Generating Figure 5: Cost breakdown waterfall...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Cost Component Breakdown – Cheapest vs Most Expensive Production Site\n(Baseline Scenario)',
             fontsize=13, fontweight='bold')

res  = results['S1_Baseline']
idx_min = res['lcoh_delivered'].idxmin()
idx_max = res['lcoh_delivered'].idxmax()

components = [
    ('c_electricity',   'Electricity\n(LCOE)',    '#FFD700'),
    ('c_electrolyser',  'Electrolyser\nCapEx',    '#4CAF50'),
    ('c_storage',       'H₂ Storage',             '#2196F3'),
    ('c_water',         'Water\nTreatment',        '#00BCD4'),
    ('c_road_infra',    'Road\nInfra.',            '#795548'),
    ('c_nh3_synth',     'NH₃\nSynthesis',         '#9C27B0'),
    ('c_road_transport','Truck to\nPort',          '#FF5722'),
    ('c_shipping',      'Ocean\nShipping',         '#607D8B'),
    ('c_port',          'Port\nHandling',          '#78909C'),
    ('c_reconversion',  'NH₃\nReconversion',       '#E91E63'),
]

for ax, (idx, title_suf) in zip(axes, [(idx_min, 'Lowest-Cost Site'),
                                        (idx_max, 'Highest-Cost Site')]):
    vals = [res[k].loc[idx] for k, _, _ in components]
    labels = [lbl for _, lbl, _ in components]
    colors = [c for _, _, c in components]
    total  = sum(vals)

    bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.7)
    ax.set_ylabel('Cost (€/kg_H₂)')
    ax.set_title(f'{title_suf}\nTotal: {total:.2f} €/kg_H₂', fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.20)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    # Value labels on bars
    for bar, val in zip(bars, vals):
        if val > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Total line
    ax.axhline(y=total, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.text(len(vals)-0.5, total + max(vals)*0.02,
            f'Total: {total:.2f}', ha='right', fontsize=10, fontweight='bold')

    # Site info
    site_info = (f"Site: {idx}\n"
                 f"Lat: {df.loc[idx,'lat']:.1f}°, Lon: {df.loc[idx,'lon']:.1f}°\n"
                 f"Solar CF: {df.loc[idx,'theo_pv']*P['solar_flh_max']/8760:.3f}\n"
                 f"Wind CF: {df.loc[idx,'theo_wind']*P['wind_flh_max']/8760:.3f}\n"
                 f"Ocean dist: {df.loc[idx,'ocean_dist_km']:.0f} km")
    ax.text(0.02, 0.97, site_info, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig05_cost_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig05_cost_breakdown.png")

# ── Fig 6: Scenario Comparison (boxplot + EU reference) ───────
print("Generating Figure 6: Scenario comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle('Scenario Comparison: Production LCOH and Delivered Cost to Rotterdam',
             fontsize=13, fontweight='bold')

sc_labels = [SCENARIOS[s]['short_label'] for s in SCENARIOS]
sc_colors  = [SCENARIOS[s]['color']      for s in SCENARIOS]

# Boxplot: production LCOH
prod_data = [results[s]['lcoh_production'].values for s in SCENARIOS]
bp1 = ax1.boxplot(prod_data, patch_artist=True, notch=False,
                   medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp1['boxes'], sc_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_xticklabels(sc_labels, rotation=15, ha='right')
ax1.set_ylabel('Production LCOH (€/kg_H₂)')
ax1.set_title('Production LCOH at African Sites')
ax1.yaxis.grid(True, alpha=0.4)
ax1.set_axisbelow(True)

# Boxplot: delivered cost + EU references
deliv_data = [results[s]['lcoh_delivered'].values for s in SCENARIOS]
bp2 = ax2.boxplot(deliv_data, patch_artist=True, notch=False,
                   medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp2['boxes'], sc_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# EU reference lines
eu_ref_prod = {s: eu_results[s]['lcoh_production'] for s in SCENARIOS}
eu_ref_dem  = {s: eu_results[s]['lcoh_at_demand']  for s in SCENARIOS}
ax2.axhline(y=np.mean(list(eu_ref_dem.values())), color='navy', linestyle='--',
            linewidth=2, label=f"EU Offshore H₂ at demand (avg {np.mean(list(eu_ref_dem.values())):.2f} €/kg)")

# Grey H₂ + carbon price reference
grey_base = grey_h2_cost_with_carbon(P, P['co2_price_baseline'])
grey_high = grey_h2_cost_with_carbon(P, P['co2_price_high'])
ax2.axhline(y=grey_base, color='brown', linestyle=':', linewidth=1.8,
            label=f"Grey H₂ (SMR) + CO₂@{P['co2_price_baseline']:.0f}€/t = {grey_base:.2f} €/kg")
ax2.axhline(y=grey_high, color='brown', linestyle='-.', linewidth=1.8,
            label=f"Grey H₂ (SMR) + CO₂@{P['co2_price_high']:.0f}€/t = {grey_high:.2f} €/kg")

ax2.set_xticklabels(sc_labels, rotation=15, ha='right')
ax2.set_ylabel('Delivered Cost (€/kg_H₂) at Rotterdam')
ax2.set_title('Delivered Cost to Rotterdam vs EU/Grey H₂ Reference')
ax2.legend(fontsize=9, loc='upper right')
ax2.yaxis.grid(True, alpha=0.4)
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig06_scenario_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig06_scenario_comparison.png")

# ── Fig 7: WACC Sensitivity ────────────────────────────────────
print("Generating Figure 7: WACC sensitivity analysis...")
waccs = np.linspace(0.04, 0.15, 12)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Sensitivity of Green H₂ Cost to Cost of Capital (WACC)',
             fontsize=13, fontweight='bold')

# Production LCOH vs WACC (best site)
idx_best = results['S1_Baseline']['lcoh_production'].idxmin()
idx_worst = results['S1_Baseline']['lcoh_production'].idxmax()

for idx, lbl, ls in [(idx_best, 'Best site (lowest LCOH)', '-'),
                      (idx_worst, 'Worst site (highest LCOH)', '--')]:
    lcoh_vals, deliv_vals = [], []
    for w in waccs:
        sc_temp = {'wacc_africa': w, 'wacc_eu': 0.05,
                   'capex_mult': 1.0, 'ship_mult': 1.0, 'reconv_mult': 1.0}
        r = lcoh_delivered(P, sc_temp)
        lcoh_vals.append(r['lcoh_production'].loc[idx])
        deliv_vals.append(r['lcoh_delivered'].loc[idx])
    axes[0].plot(waccs*100, lcoh_vals, linestyle=ls, linewidth=2,
                 label=lbl, marker='o', markersize=5)
    axes[1].plot(waccs*100, deliv_vals, linestyle=ls, linewidth=2,
                 label=lbl, marker='o', markersize=5)

# EU reference bands
eu_prod_range = [eu_results[s]['lcoh_production'] for s in SCENARIOS]
eu_dem_range  = [eu_results[s]['lcoh_at_demand']  for s in SCENARIOS]
axes[0].axhspan(min(eu_prod_range), max(eu_prod_range), alpha=0.2, color='navy',
                label=f'EU offshore H₂ LCOH range ({min(eu_prod_range):.1f}–{max(eu_prod_range):.1f} €/kg)')
axes[1].axhspan(min(eu_dem_range), max(eu_dem_range), alpha=0.2, color='navy',
                label=f'EU H₂ at demand range ({min(eu_dem_range):.1f}–{max(eu_dem_range):.1f} €/kg)')
axes[1].axhline(y=grey_base, color='brown', linestyle=':', linewidth=1.5,
                label=f'Grey H₂ + CO₂ ({grey_base:.2f} €/kg)')

for ax, title in [(axes[0], 'Production LCOH (Africa)'),
                   (axes[1], 'Delivered LCOH (at Rotterdam)')]:
    ax.set_xlabel('WACC in Africa (%)')
    ax.set_ylabel('Cost (€/kg_H₂)')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

# Mark scenario WACC levels
for sc_name, sc in SCENARIOS.items():
    axes[0].axvline(x=sc['wacc_africa']*100, color=sc['color'],
                    linestyle=':', alpha=0.7, linewidth=1)
    axes[1].axvline(x=sc['wacc_africa']*100, color=sc['color'],
                    linestyle=':', alpha=0.7, linewidth=1,
                    label=f"_{sc['short_label']}")

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig07_wacc_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig07_wacc_sensitivity.png")

# ── Fig 8: Supply Chain Waterfall (best site, baseline) ───────
print("Generating Figure 8: Supply chain waterfall...")
res  = results['S1_Baseline']
idx_min = res['lcoh_delivered'].idxmin()

stages = [
    ('Electricity\n(LCOE)',   res['c_electricity'].loc[idx_min]),
    ('Electrolyser',           res['c_electrolyser'].loc[idx_min]),
    ('H₂ Storage',             res['c_storage'].loc[idx_min]),
    ('Water',                  res['c_water'].loc[idx_min]),
    ('Road Infra.',            res['c_road_infra'].loc[idx_min]),
    ('NH₃ Synthesis',          res['c_nh3_synth'].loc[idx_min]),
    ('Truck→Port',             res['c_road_transport'].loc[idx_min]),
    ('NH₃ Shipping',           float(res['c_shipping'].iloc[0])),
    ('Port Handling',          float(res['c_port'].iloc[0])),
    ('Reconversion',           float(res['c_reconversion'].iloc[0])),
]

labels = [s[0] for s in stages]
values = [s[1] for s in stages]
cum    = np.cumsum([0] + values[:-1])

colors_wf = ['#FFD700','#4CAF50','#2196F3','#00BCD4','#795548',
              '#9C27B0','#FF5722','#607D8B','#78909C','#E91E63']

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(labels, values, bottom=cum, color=colors_wf,
              edgecolor='white', linewidth=0.7, width=0.6)

# Cumulative line
total_cum = cum + np.array(values)
ax.step(range(len(labels)), total_cum, where='post',
        color='black', linewidth=2.0, linestyle='--', label='Cumulative cost')
ax.scatter(range(len(labels)), total_cum,
           color='black', s=60, zorder=5)

ax.set_ylabel('Cost (€/kg_H₂)', fontsize=12)
ax.set_title(f'Supply Chain Cost Breakdown – Lowest-Cost Site (Baseline Scenario)\n'
             f'Site: {idx_min} | Lat: {df.loc[idx_min,"lat"]:.1f}° | '
             f'Lon: {df.loc[idx_min,"lon"]:.1f}° | '
             f'Total delivered: {sum(values):.2f} €/kg_H₂',
             fontweight='bold')
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
ax.legend(fontsize=11)
ax.set_ylim(0, sum(values) * 1.15)

# Value labels
for i, (bar, val, bottom) in enumerate(zip(bars, values, cum)):
    ax.text(bar.get_x() + bar.get_width()/2, bottom + val/2,
            f'{val:.2f}', ha='center', va='center', fontsize=9,
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.4))

# Add divider between production and supply chain
ax.axvline(x=4.5, color='grey', linestyle=':', linewidth=1.5)
ax.text(2.0, sum(values)*1.07, '◀ Production ▶', ha='center', fontsize=9,
        color='grey', fontweight='bold')
ax.text(7.0, sum(values)*1.07, '◀ Supply chain to Europe ▶', ha='center', fontsize=9,
        color='grey', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig08_supply_chain_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig08_supply_chain_waterfall.png")

# ── Fig 9: Cost Curves (sorted) ───────────────────────────────
print("Generating Figure 9: Cost supply curves...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Green H₂ Supply Cost Curves (All Hexagons, Sorted by Cost)',
             fontsize=13, fontweight='bold')

for sc_name, sc in SCENARIOS.items():
    sorted_prod = np.sort(results[sc_name]['lcoh_production'].values)
    sorted_deliv = np.sort(results[sc_name]['lcoh_delivered'].values)
    x = np.arange(1, len(sorted_prod)+1)
    ax1.plot(x, sorted_prod, color=sc['color'], linewidth=2.5,
             linestyle=sc['linestyle'], label=sc['short_label'])
    ax2.plot(x, sorted_deliv, color=sc['color'], linewidth=2.5,
             linestyle=sc['linestyle'], label=sc['short_label'])

# EU reference
eu_vals = [eu_results[s]['lcoh_at_demand'] for s in SCENARIOS]
ax2.axhline(y=np.mean(eu_vals), color='navy', linestyle='--', linewidth=2,
            label=f'EU H₂ at demand (avg {np.mean(eu_vals):.2f} €/kg)')
ax2.axhline(y=grey_base, color='brown', linestyle=':', linewidth=1.8,
            label=f'Grey H₂+CO₂ ({grey_base:.2f} €/kg)')

for ax, title, ylabel in [
    (ax1, 'Production LCOH (at African site)', 'LCOH (€/kg_H₂)'),
    (ax2, 'Delivered LCOH at Rotterdam',       'Delivered LCOH (€/kg_H₂)'),
]:
    ax.set_xlabel('Hexagon rank (cheapest → most expensive)', fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(0.5, len(df)+0.5)

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig09_cost_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig09_cost_curves.png")

# ── Fig 10: Competitiveness Map (vs EU offshore H₂) ──────────
print("Generating Figure 10: Competitiveness map...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Cost Competitiveness vs. European Offshore Wind H₂\n'
             '(Green = African H₂ cheaper at Rotterdam, Red = EU H₂ cheaper)',
             fontsize=13, fontweight='bold')

for ax, (sc_name, sc) in zip(axes.flat, SCENARIOS.items()):
    res = results[sc_name]
    eu  = eu_results[sc_name]
    advantage = eu['lcoh_at_demand'] - res['lcoh_delivered']   # + = Africa cheaper

    # Diverging colormap centred on zero
    vext = max(abs(advantage.min()), abs(advantage.max()))
    norm = mcolors.TwoSlopeNorm(vmin=-vext, vcenter=0, vmax=vext)

    gdf_africa.plot(ax=ax, color='#f0f0eb', edgecolor='grey', linewidth=0.4)
    sc_plot = ax.scatter(df['lon'], df['lat'], c=advantage, cmap='RdYlGn',
                         norm=norm, s=160, zorder=5,
                         edgecolors='white', linewidths=0.3)
    cb = plt.colorbar(sc_plot, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label('Cost advantage of Africa (€/kg_H₂)\n+ve = Africa cheaper', size=9)
    n_cheaper = (advantage > 0).sum()
    ax.set_title(f"{sc['short_label']}\n"
                 f"EU ref: {eu['lcoh_at_demand']:.2f} €/kg | "
                 f"{n_cheaper}/{len(df)} sites competitive",
                 fontweight='bold')
    ax.set_xlim(df['lon'].min()-3, df['lon'].max()+3)
    ax.set_ylim(df['lat'].min()-3, df['lat'].max()+3)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig10_competitiveness_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig10_competitiveness_map.png")

# ── Fig 11: Carbon Price Threshold Analysis ────────────────────
print("Generating Figure 11: Carbon price threshold...")
co2_prices = np.linspace(0, 300, 50)

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_title('Break-Even CO₂ Price for African vs. Grey H₂ Competitiveness',
             fontweight='bold', fontsize=13)

for sc_name, sc in SCENARIOS.items():
    deliv_min  = results[sc_name]['lcoh_delivered'].min()
    deliv_med  = results[sc_name]['lcoh_delivered'].median()
    grey_costs = [grey_h2_cost_with_carbon(P, co2) for co2 in co2_prices]

    # Shaded band from min to median
    grey_arr = np.array(grey_costs)
    ax.plot(co2_prices, grey_arr, color='brown', linestyle='-',
            linewidth=2, zorder=3, label='Grey H₂ (SMR) cost' if sc_name == 'S1_Baseline' else '_nolegend_')
    ax.axhline(y=deliv_min, color=sc['color'], linestyle=sc['linestyle'],
               linewidth=2, label=f"{sc['short_label']} – best site ({deliv_min:.2f} €/kg)")
    ax.axhline(y=deliv_med, color=sc['color'], linestyle='--',
               linewidth=1.5, alpha=0.6,
               label=f"{sc['short_label']} – median ({deliv_med:.2f} €/kg)")

ax.set_xlabel('CO₂ Price (€/tCO₂)', fontsize=12)
ax.set_ylabel('H₂ Cost (€/kg_H₂)', fontsize=12)
ax.legend(fontsize=8.5, loc='upper left', ncol=1)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
ax.set_xlim(0, 300)

# Mark baseline CO₂ price
ax.axvline(x=P['co2_price_baseline'], color='grey', linestyle=':', linewidth=1.5,
           label=f"Baseline CO₂ price ({P['co2_price_baseline']} €/t)")
ax.axvline(x=P['co2_price_high'], color='grey', linestyle='-.', linewidth=1.5,
           label=f"High CO₂ price ({P['co2_price_high']} €/t)")

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig11_carbon_price_threshold.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig11_carbon_price_threshold.png")

# ── Fig 12: Stacked bar – all scenarios summary ───────────────
print("Generating Figure 12: Stacked scenario summary...")
fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle('Cost Component Stacking – Median Delivered H₂ Cost by Scenario',
             fontweight='bold', fontsize=13)

comp_keys = [
    ('c_electricity',   'Electricity (LCOE)',    '#FFD700'),
    ('c_electrolyser',  'Electrolyser CAPEX',    '#4CAF50'),
    ('c_storage',       'H₂ Storage',            '#2196F3'),
    ('c_water',         'Water',                 '#00BCD4'),
    ('c_road_infra',    'Road Infra.',            '#795548'),
    ('c_nh3_synth',     'NH₃ Synthesis',         '#9C27B0'),
    ('c_road_transport','Truck→Port',             '#FF5722'),
    ('c_shipping',      'Shipping',               '#607D8B'),
    ('c_port',          'Port Handling',          '#78909C'),
    ('c_reconversion',  'Reconversion',           '#E91E63'),
]

sc_names = list(SCENARIOS.keys())
x = np.arange(len(sc_names))
bar_width = 0.55
bottoms = np.zeros(len(sc_names))
patches = []

for key, label, color in comp_keys:
    vals = []
    for sc_name in sc_names:
        v = results[sc_name][key]
        if hasattr(v, 'median'):
            vals.append(v.median())
        else:
            vals.append(float(v.iloc[0]) if hasattr(v,'iloc') else float(v))
    bars = ax.bar(x, vals, bar_width, bottom=bottoms, color=color,
                  edgecolor='white', linewidth=0.5, label=label)
    bottoms += np.array(vals)
    patches.append(Patch(facecolor=color, label=label))

# EU reference line
eu_med = np.mean([eu_results[s]['lcoh_at_demand'] for s in SCENARIOS])
ax.axhline(y=eu_med, color='navy', linestyle='--', linewidth=2,
           label=f'EU offshore H₂ at demand (avg: {eu_med:.2f} €/kg)')

ax.set_xticks(x)
ax.set_xticklabels([SCENARIOS[s]['short_label'] for s in sc_names], fontsize=11)
ax.set_ylabel('Median Delivered Cost (€/kg_H₂)', fontsize=12)
ax.set_title('Median Delivered H₂ Cost Breakdown by Scenario', fontweight='bold')
ax.legend(handles=patches + [Line2D([0],[0], color='navy', linestyle='--',
                                     label=f'EU ref: {eu_med:.2f} €/kg')],
          loc='upper right', fontsize=9, ncol=2)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig12_stacked_scenario_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig12_stacked_scenario_summary.png")

# ── Fig 13: WACC effect – production vs supply chain split ────
print("Generating Figure 13: WACC effect on production vs supply chain costs...")
waccs_fine = np.linspace(0.04, 0.14, 20)
best_idx   = results['S1_Baseline']['lcoh_delivered'].idxmin()

prod_costs, sc_costs = [], []
for w in waccs_fine:
    sc_temp = {'wacc_africa': w, 'wacc_eu': 0.05,
               'capex_mult': 1.0, 'ship_mult': 1.0, 'reconv_mult': 1.0}
    r = lcoh_delivered(P, sc_temp)
    prod_costs.append(r['lcoh_production'].loc[best_idx])
    sc_costs.append(r['lcoh_delivered'].loc[best_idx] - r['lcoh_production'].loc[best_idx])

prod_arr = np.array(prod_costs)
sc_arr   = np.array(sc_costs)

fig, ax = plt.subplots(figsize=(10, 6))
ax.stackplot(waccs_fine*100, prod_arr, sc_arr,
             labels=['Production cost (at site)', 'Supply chain to Rotterdam'],
             colors=['#4CAF50', '#2196F3'], alpha=0.8)
ax.set_xlabel('WACC in Africa (%)', fontsize=12)
ax.set_ylabel('Delivered Cost (€/kg_H₂)', fontsize=12)
ax.set_title(f'Effect of WACC on Production vs Supply Chain Costs\n'
             f'(Best site: {best_idx}, Baseline Scenario)',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=11)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)

# Mark scenario WACCs
for sc_name, sc in SCENARIOS.items():
    ax.axvline(x=sc['wacc_africa']*100, color=sc['color'], linestyle='--',
               linewidth=1.5, alpha=0.8, label=sc['short_label'])
ax.legend(fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig13_wacc_prod_vs_sc.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig13_wacc_prod_vs_sc.png")

print("\n✓ All figures saved to", IMAGES_DIR)
print("✓ All outputs saved to", OUTPUT_DIR)
print("\n=== ANALYSIS COMPLETE ===")
