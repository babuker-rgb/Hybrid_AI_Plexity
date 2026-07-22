"""
Hubryd AI – v29.27-R31 (FINAL FIX – No Build Function Conflict)
Hybrid AI For Multi-Objective Tablet Optimization
Nile Valley University, Sudan
"""

import streamlit as st

# ================================================================
# MUST BE FIRST STREAMLIT COMMAND
# ================================================================
st.set_page_config(page_title="Hybrid AI · Tablet Optimization", layout="wide")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# ================================================================
# CONSTANTS
# ================================================================
D_MIN = 0.72
D_MAX = 0.99
TENSILE_MIN = 1.50
EFRF_MAX = 0.50
DISINTEGRATION_MAX = 15.0

SLIDER_API_MIN = 80.0
SLIDER_API_MAX = 98.0
SLIDER_MCC_MIN = 1.5
SLIDER_MCC_MAX = 8.0
SLIDER_PVPP_MIN = 1.0
SLIDER_PVPP_MAX = 6.0
SLIDER_MGST_MIN = 0.10
SLIDER_MGST_MAX = 1.2
SLIDER_BINDER_MIN = 1.4
SLIDER_BINDER_MAX = 6.0
SLIDER_MOISTURE_MIN = 0.5
SLIDER_MOISTURE_MAX = 5.0

BINDER_GRADES = ["MCC PH101", "MCC PH102", "MCC PH200", "MCC KG", "Lactose", "Dicalcium Phosphate"]

SLIDER_PRESSURE_MIN = 150.0
SLIDER_PRESSURE_MAX = 250.0
SLIDER_SPEED_MIN = 15.0
SLIDER_SPEED_MAX = 30.0
SLIDER_GRANULE_MIN = 30.0
SLIDER_GRANULE_MAX = 250.0
SLIDER_DWELL_TIME_MIN = 5.0
SLIDER_DWELL_TIME_MAX = 50.0
SLIDER_FRICTION_MIN = 0.1
SLIDER_FRICTION_MAX = 0.5
SLIDER_DECOMPRESSION_TIME_MIN = 10.0
SLIDER_DECOMPRESSION_TIME_MAX = 80.0
SLIDER_PARTICLE_SIZE_MIN = 10.0
SLIDER_PARTICLE_SIZE_MAX = 200.0

BOUND_MCC_MIN = 2.0
BOUND_MCC_MAX = 8.0
BOUND_PVPP_MIN = 1.5
BOUND_PVPP_MAX = 6.0
BOUND_MGST_MIN = 0.3
BOUND_MGST_MAX = 1.2
BOUND_BINDER_MIN = 3.0
BOUND_BINDER_MAX = 6.0
BOUND_PRESSURE_MIN = 150.0
BOUND_PRESSURE_MAX = 250.0
BOUND_SPEED_MIN = 15.0
BOUND_SPEED_MAX = 30.0
BOUND_GRANULE_MIN = 30.0
BOUND_GRANULE_MAX = 250.0

# Training parameters
N_SAMPLES = 12000
ADAM_EPOCHS = 400
PATIENCE = 60
NSGA_POP = 80
NSGA_GENS = 50
HIDDEN_SIZE = 256

# Loss weights
W_DENSITY = 1.0
W_TENSILE = 500.0
W_ER = 5.0
W_PHYSICS = 0.0
W_EFRF_PENALTY = 0.0
W_DISINTEGRATION = 50.0
W_DISSOLUTION = 20.0

# ================================================================
# SESSION STATE
# ================================================================
if 'api' not in st.session_state:
    st.session_state.update({
        'api': 90.5,
        'binder': 3.5,
        'pvpp': 2.0,
        'mgst': 0.5,
        'mcc': 3.5,
        'moisture': 2.0,
        'particle_size': 50.0,
        'binder_grade': 0,
        'pressure': 200.0,
        'speed': 20.0,
        'dwell_time': 25.0,
        'friction': 0.25,
        'decompression_time': 35.0,
        'granule': 125.0,
        'show_cost_solution': False,
        'show_quality_solution': False,
        'show_comparison': False,
        'show_sensitivity': False,
        'show_dissolution': False,
        'granule_mode': 'Fixed',
        'nsga_pop': None,
        'nsga_objectives': None,
        'nsga_fronts': None,
        'balanced_solution': None,
        'quality_solution': None,
        'cost_solution': None,
        'run_optimized': False,
        'formulation': None,
        'feasible_df': None,
        'tested_point': None,
        'benchmark_df': None,
        'experimental_data': None
    })

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def normalize_components(api, binder, pvpp, mgst, mcc, moisture):
    api = np.asarray(api, dtype=float)
    binder = np.asarray(binder, dtype=float)
    pvpp = np.asarray(pvpp, dtype=float)
    mgst = np.asarray(mgst, dtype=float)
    mcc = np.asarray(mcc, dtype=float)
    moisture = np.asarray(moisture, dtype=float)

    api = np.clip(api, SLIDER_API_MIN, SLIDER_API_MAX)
    binder = np.clip(binder, SLIDER_BINDER_MIN, SLIDER_BINDER_MAX)
    pvpp = np.clip(pvpp, SLIDER_PVPP_MIN, SLIDER_PVPP_MAX)
    mgst = np.clip(mgst, SLIDER_MGST_MIN, SLIDER_MGST_MAX)
    mcc = np.clip(mcc, SLIDER_MCC_MIN, SLIDER_MCC_MAX)
    moisture = np.clip(moisture, SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX)

    total = api + binder + pvpp + mgst + mcc + moisture
    total = np.where(total <= 0, 1.0, total)

    api = (api / total) * 100.0
    binder = (binder / total) * 100.0
    pvpp = (pvpp / total) * 100.0
    mgst = (mgst / total) * 100.0
    mcc = (mcc / total) * 100.0
    moisture = (moisture / total) * 100.0

    api = np.clip(api, SLIDER_API_MIN, SLIDER_API_MAX)
    binder = np.clip(binder, SLIDER_BINDER_MIN, SLIDER_BINDER_MAX)
    pvpp = np.clip(pvpp, SLIDER_PVPP_MIN, SLIDER_PVPP_MAX)
    mgst = np.clip(mgst, SLIDER_MGST_MIN, SLIDER_MGST_MAX)
    mcc = np.clip(mcc, SLIDER_MCC_MIN, SLIDER_MCC_MAX)
    moisture = np.clip(moisture, SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX)

    total2 = api + binder + pvpp + mgst + mcc + moisture
    total2 = np.where(total2 <= 0, 1.0, total2)
    scale = 100.0 / total2
    api = api * scale
    binder = binder * scale
    pvpp = pvpp * scale
    mgst = mgst * scale
    mcc = mcc * scale
    moisture = moisture * scale

    api = np.clip(api, SLIDER_API_MIN, SLIDER_API_MAX)
    binder = np.clip(binder, SLIDER_BINDER_MIN, SLIDER_BINDER_MAX)
    pvpp = np.clip(pvpp, SLIDER_PVPP_MIN, SLIDER_PVPP_MAX)
    mgst = np.clip(mgst, SLIDER_MGST_MIN, SLIDER_MGST_MAX)
    mcc = np.clip(mcc, SLIDER_MCC_MIN, SLIDER_MCC_MAX)
    moisture = np.clip(moisture, SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX)

    return api, binder, pvpp, mgst, mcc, moisture

def calculate_dwell_time(speed_rpm, punch_width=10, pitch_diameter=100):
    if np.isscalar(speed_rpm):
        speed_rpm = np.array([speed_rpm])
    speed_rpm = np.asarray(speed_rpm)
    result = np.full_like(speed_rpm, 50.0, dtype=float)
    mask = speed_rpm > 0
    result[mask] = (punch_width * 60 * 1000) / (np.pi * pitch_diameter * speed_rpm[mask])
    result = np.clip(result, 5.0, 80.0)
    return result

def predict_disintegration_time(tensile, pvpp_n, api_n, binder_n, moisture_n):
    tensile = np.asarray(tensile)
    pvpp_n = np.asarray(pvpp_n)
    api_n = np.asarray(api_n)
    binder_n = np.asarray(binder_n)
    moisture_n = np.asarray(moisture_n)

    base_time = 2.0 + 0.5 * tensile
    pvpp_effect = 5.0 * np.exp(-0.5 * pvpp_n)
    api_effect = 0.1 * (api_n - 80)
    binder_effect = 0.2 * (binder_n - 2.0)
    moisture_effect = -0.1 * moisture_n
    time = base_time - pvpp_effect + api_effect + binder_effect + moisture_effect
    return np.clip(time, 1.0, 30.0)

def predict_dissolution_profile(api_n, pvpp_n, particle_size, disintegration_time):
    api_n = np.asarray(api_n)
    pvpp_n = np.asarray(pvpp_n)
    particle_size = np.asarray(particle_size)
    disintegration_time = np.asarray(disintegration_time)

    tau = 5.0 + 0.5 * disintegration_time - 0.1 * pvpp_n + 0.05 * (api_n - 80)
    tau = np.clip(tau, 2.0, 20.0)
    beta = 1.0 + 0.01 * (particle_size - 50) / 50
    beta = np.clip(beta, 0.8, 2.5)
    return {'tau': tau, 'beta': beta}

# ================================================================
# DATA GENERATION – with features pre-built
# ================================================================

def generate_pinn_data_with_features(n_samples=N_SAMPLES, random_state=42):
    rng = np.random.default_rng(random_state)

    api_raw = rng.uniform(SLIDER_API_MIN, SLIDER_API_MAX, n_samples)
    binder_raw = rng.uniform(SLIDER_BINDER_MIN, SLIDER_BINDER_MAX, n_samples)
    pvpp_raw = rng.uniform(SLIDER_PVPP_MIN, SLIDER_PVPP_MAX, n_samples)
    mgst_raw = rng.uniform(SLIDER_MGST_MIN, SLIDER_MGST_MAX, n_samples)
    mcc_raw = rng.uniform(SLIDER_MCC_MIN, SLIDER_MCC_MAX, n_samples)
    moisture_raw = rng.uniform(SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX, n_samples)

    api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n = normalize_components(
        api_raw, binder_raw, pvpp_raw, mgst_raw, mcc_raw, moisture_raw
    )

    particle_size_raw = rng.uniform(SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX, n_samples)
    binder_grade_raw = rng.integers(0, len(BINDER_GRADES), n_samples)
    pressure_raw = rng.uniform(SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX, n_samples)
    speed_raw = rng.uniform(SLIDER_SPEED_MIN, SLIDER_SPEED_MAX, n_samples)
    dwell_time_raw = calculate_dwell_time(speed_raw)
    friction_raw = rng.uniform(SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX, n_samples)
    decompression_time_raw = rng.uniform(SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX, n_samples)
    granule_raw = rng.uniform(SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX, n_samples)

    X_raw = np.column_stack([
        api_n, mcc_n, pvpp_n, mgst_n, binder_n,
        pressure_raw, speed_raw, granule_raw,
        particle_size_raw, moisture_n, binder_grade_raw,
        dwell_time_raw, friction_raw, decompression_time_raw
    ])

    # Density: Physical model
    k_heckel = 0.025 + 0.0001 * pressure_raw
    A_heckel = 1.0 + 0.01 * (api_n - 85.0) - 0.05 * binder_n
    x_val = k_heckel * pressure_raw + A_heckel
    D_heckel = 1.0 - np.exp(-x_val)
    D_heckel = np.clip(D_heckel, D_MIN, D_MAX)

    a_kawakita = 0.82 + 0.04 * (mcc_n - 1.5) / 6.5 + 0.02 * (binder_n - 1.4) / 4.6
    a_kawakita = np.clip(a_kawakita, 0.78, 0.92)
    b_kawakita = 0.002 + 0.003 * (binder_n - 1.4) / 4.6 + 0.001 * (mcc_n - 1.5) / 6.5
    b_kawakita = np.clip(b_kawakita, 0.0005, 0.006)
    D_kawakita = 1.0 - pressure_raw / (a_kawakita * pressure_raw + 1.0 / b_kawakita)
    D_kawakita = np.clip(D_kawakita, D_MIN, D_MAX)

    pressure_norm = (pressure_raw - SLIDER_PRESSURE_MIN) / (SLIDER_PRESSURE_MAX - SLIDER_PRESSURE_MIN)
    w_heckel = pressure_norm
    w_kawakita = 1.0 - pressure_norm

    D = w_heckel * D_heckel + w_kawakita * D_kawakita

    moisture_effect = -0.003 * (moisture_n - 2.0)
    moisture_effect = np.clip(moisture_effect, -0.02, 0.01)
    particle_effect = -0.002 * (particle_size_raw - 50) / 150
    particle_effect = np.clip(particle_effect, -0.02, 0.01)
    speed_effect = -0.002 * (speed_raw - 15) / 15
    speed_effect = np.clip(speed_effect, -0.015, 0.0)
    mgst_effect = -0.01 * (mgst_n - 0.2)
    mgst_effect = np.clip(mgst_effect, -0.02, 0.005)

    D += moisture_effect + particle_effect + speed_effect + mgst_effect
    D = np.clip(D, D_MIN, D_MAX)

    # Tensile Strength
    porosity = 1.0 - D
    sigma0 = 5.0 + 0.1 * (api_n - 85.0) + 0.2 * binder_n - 0.5 * mgst_n
    sigma0 = np.clip(sigma0, 2.0, 8.0)
    b = 2.5 - 0.005 * (pressure_raw - 80.0) - 0.01 * (particle_size_raw - 50) / 100
    b = np.clip(b, 1.5, 3.5)

    tensile_base = sigma0 * np.exp(-b * porosity)
    api_effect = 1.0 - 0.005 * (api_n - 85.0)
    binder_effect = 1.0 + 0.03 * (binder_n - 2.0)
    mgst_effect = 1.0 - 0.1 * (mgst_n - 0.2)
    pvpp_effect = 1.0 - 0.02 * (pvpp_n - 3.0)
    speed_effect = 1.0 - 0.002 * (speed_raw - 10.0)
    particle_effect = 1.0 - 0.0005 * (particle_size_raw - 50)
    particle_effect = np.clip(particle_effect, 0.8, 1.2)

    strength = (tensile_base * api_effect * binder_effect *
                mgst_effect * pvpp_effect * speed_effect * particle_effect)
    strength = np.clip(strength, 0.5, 6.0)

    # Elastic Recovery
    er_base = (1.8 + 0.3 * (api_n - 85.0)/10.0 +
               0.08 * (speed_raw - 10.0)/30.0 -
               0.1 * (pressure_raw - 100.0)/150.0 +
               0.02 * (decompression_time_raw - 35.0)/30.0)
    er_base = er_base * (1.0 - 0.15 * (D - 0.4))
    er = np.clip(er_base, 0.5, 4.0)

    # Disintegration & Dissolution
    disintegration = predict_disintegration_time(strength, pvpp_n, api_n, binder_n, moisture_n)
    disintegration = np.clip(disintegration, 1.0, 30.0)

    dissolution_params = predict_dissolution_profile(api_n, pvpp_n, particle_size_raw, disintegration)
    dissolution_tau = np.clip(dissolution_params['tau'], 2.0, 20.0)
    dissolution_beta = np.clip(dissolution_params['beta'], 0.8, 2.5)

    # Create DataFrame with features
    df = pd.DataFrame(X_raw, columns=[
        'API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%',
        'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm',
        'Particle_Size_µm', 'Moisture_%', 'Binder_Grade',
        'Dwell_Time_ms', 'Friction', 'Decompression_Time_ms'
    ])
    df['Density'] = D
    df['Tensile_Strength_MPa'] = strength
    df['Elastic_Recovery_%'] = er
    df['Disintegration_Time_min'] = disintegration
    df['Dissolution_Tau'] = dissolution_tau
    df['Dissolution_Beta'] = dissolution_beta

    return df

# ================================================================
# BUILD FEATURES FOR PREDICTION (ONLY HERE)
# ================================================================

def build_prediction_features(X_raw):
    """Build features for prediction - only called during predict_pinn"""
    if X_raw.ndim == 1:
        X_raw = X_raw.reshape(1, -1)
    
    # Extract columns
    api = X_raw[:, 0:1]
    mcc = X_raw[:, 1:2]
    pvpp = X_raw[:, 2:3]
    mgst = X_raw[:, 3:4]
    binder = X_raw[:, 4:5]
    pressure = X_raw[:, 5:6]
    speed = X_raw[:, 6:7]
    particle_size = X_raw[:, 8:9]
    moisture = X_raw[:, 9:10]
    dwell_time = X_raw[:, 11:12]
    friction = X_raw[:, 12:13]
    
    # 17 interaction features
    pb = pressure * binder
    pa = pressure * api
    ps = np.clip(pressure / (speed + 0.1), 0, 1000)
    am = np.clip(api / (mcc + 0.1), 0, 1000)
    bs = np.clip(binder / (speed + 0.1), 0, 100)
    ap = api * pvpp
    bm = binder * mgst
    mp = mcc * pvpp
    a2 = api ** 2
    p2 = pressure ** 2
    b2 = binder ** 2
    s2 = speed ** 2
    pp = particle_size * pressure
    mp2 = moisture * pressure
    pm = particle_size * moisture
    dp = dwell_time * pressure
    fp = friction * pressure
    
    return np.concatenate([
        X_raw,
        pb, pa, ps, am, bs, ap, bm, mp,
        a2, p2, b2, s2, pp, mp2, pm, dp, fp
    ], axis=1)

# ================================================================
# PINN MODEL
# ================================================================

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class ResidualBlock(nn.Module):
    def __init__(self, features, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(features, features)
        self.bn1 = nn.BatchNorm1d(features)
        self.lin2 = nn.Linear(features, features)
        self.bn2 = nn.BatchNorm1d(features)
        self.act = Mish()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.lin1(x)))
        out = self.drop(out)
        out = self.bn2(self.lin2(out))
        out = self.drop(out)
        return identity + out

class MultiTaskPINN(nn.Module):
    def __init__(self, input_dim, hidden=HIDDEN_SIZE):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden), Mish(), nn.Dropout(0.05))
        self.res1 = ResidualBlock(hidden, dropout=0.05)
        self.res2 = ResidualBlock(hidden, dropout=0.05)
        self.transition = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.Tanh(),
            nn.Dropout(0.05)
        )
        self.output = nn.Linear(hidden//2, 10)

    def forward(self, X):
        x = self.input_layer(X)
        x = self.res1(x)
        x = self.res2(x)
        x = self.transition(x)
        raw = self.output(x)

        density = raw[:, 0:1]
        tensile = raw[:, 1:2]
        er = raw[:, 2:3]
        k_heckel = torch.nn.functional.softplus(raw[:, 3:4]) + 1e-4
        A_heckel = torch.nn.functional.softplus(raw[:, 4:5]) + 1e-4
        a_kawakita = torch.nn.functional.softplus(raw[:, 5:6]) + 1e-4
        b_kawakita = torch.nn.functional.softplus(raw[:, 6:7]) + 1e-4
        disintegration = torch.nn.functional.softplus(raw[:, 7:8])
        dissolution_tau = torch.nn.functional.softplus(raw[:, 8:9])
        dissolution_beta = torch.nn.functional.softplus(raw[:, 9:10]) + 1e-4

        return torch.cat([density, tensile, er,
                          k_heckel, A_heckel, a_kawakita, b_kawakita,
                          disintegration, dissolution_tau, dissolution_beta], dim=1)

    def predict(self, X_scaled):
        self.eval()
        with torch.no_grad():
            if not isinstance(X_scaled, torch.Tensor):
                X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
            device = next(self.parameters()).device
            X_scaled = X_scaled.to(device)
            output = self.forward(X_scaled)
            selected = torch.cat([output[:, 0:3], output[:, 7:10]], dim=1)
            return selected.cpu().numpy()

    def compute_loss(self, X_scaled, X_raw, y_true, y_scaler, epoch=0, total_epochs=ADAM_EPOCHS):
        y_pred = self.forward(X_scaled)
        density_pred = y_pred[:, 0:1]
        tensile_pred = y_pred[:, 1:2]
        er_pred = y_pred[:, 2:3]
        disintegration_pred = y_pred[:, 7:8]
        dissolution_tau_pred = y_pred[:, 8:9]
        dissolution_beta_pred = y_pred[:, 9:10]

        loss_dens = nn.MSELoss()(density_pred, y_true[:, 0:1])
        loss_tensile = nn.MSELoss()(tensile_pred, y_true[:, 1:2])
        loss_er = nn.MSELoss()(er_pred, y_true[:, 2:3])
        loss_disin = nn.MSELoss()(disintegration_pred, y_true[:, 3:4])
        loss_tau = nn.MSELoss()(dissolution_tau_pred, y_true[:, 4:5])
        loss_beta = nn.MSELoss()(dissolution_beta_pred, y_true[:, 5:6])

        data_loss = (W_DENSITY * loss_dens + W_TENSILE * loss_tensile + W_ER * loss_er +
                     W_DISINTEGRATION * loss_disin + W_DISSOLUTION * (loss_tau + loss_beta))

        return data_loss

# ================================================================
# NSGA-II
# ================================================================

class NSGAII:
    def __init__(self, model, scaler, y_scaler, bounds, pop=NSGA_POP, gens=NSGA_GENS,
                 granule_fixed=True, granule_fixed_val=125.0):
        self.model = model
        self.scaler = scaler
        self.y_scaler = y_scaler
        self.bounds = bounds
        self.pop_size = pop
        self.generations = gens
        self.granule_fixed = granule_fixed
        self.granule_fixed_val = granule_fixed_val

    def _repair(self, ind):
        api, mcc, pvpp, mgst, binder, pressure, speed, granule, particle_size, moisture, binder_grade, dwell_time, friction, decompression_time = ind
        api, binder, pvpp, mgst, mcc, moisture = normalize_components(
            api, binder, pvpp, mgst, mcc, moisture
        )
        pressure = np.clip(pressure, self.bounds[5,0], self.bounds[5,1])
        speed = np.clip(speed, self.bounds[6,0], self.bounds[6,1])
        particle_size = np.clip(particle_size, SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX)
        binder_grade = np.clip(binder_grade, 0, len(BINDER_GRADES)-1)
        dwell_time = np.clip(dwell_time, SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX)
        friction = np.clip(friction, SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX)
        decompression_time = np.clip(decompression_time, SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX)
        if self.granule_fixed:
            granule = self.granule_fixed_val
        else:
            granule = np.clip(granule, self.bounds[7,0], self.bounds[7,1])
        return np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule,
                         particle_size, moisture, binder_grade, dwell_time, friction, decompression_time])

    def _repair_batch(self, pop):
        api = pop[:, 0]; mcc = pop[:, 1]; pvpp = pop[:, 2]
        mgst = pop[:, 3]; binder = pop[:, 4]
        pressure = pop[:, 5]; speed = pop[:, 6]; granule = pop[:, 7]
        particle_size = pop[:, 8]; moisture = pop[:, 9]; binder_grade = pop[:, 10]
        dwell_time = pop[:, 11]; friction = pop[:, 12]; decompression_time = pop[:, 13]

        api, binder, pvpp, mgst, mcc, moisture = normalize_components(
            api, binder, pvpp, mgst, mcc, moisture
        )
        pressure = np.clip(pressure, self.bounds[5,0], self.bounds[5,1])
        speed = np.clip(speed, self.bounds[6,0], self.bounds[6,1])
        particle_size = np.clip(particle_size, SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX)
        binder_grade = np.clip(binder_grade, 0, len(BINDER_GRADES)-1)
        dwell_time = np.clip(dwell_time, SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX)
        friction = np.clip(friction, SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX)
        decompression_time = np.clip(decompression_time, SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX)
        if self.granule_fixed:
            granule = np.full_like(granule, self.granule_fixed_val)
        else:
            granule = np.clip(granule, self.bounds[7,0], self.bounds[7,1])
        return np.column_stack([api, mcc, pvpp, mgst, binder, pressure, speed, granule,
                                particle_size, moisture, binder_grade, dwell_time, friction, decompression_time])

    def _evaluate(self, population):
        n = population.shape[0]
        repaired = self._repair_batch(population)
        # Build features for evaluation
        aug = build_prediction_features(repaired)
        scaled = self.scaler.transform(aug)
        X_t = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = self.model.predict(X_t)
            pred = self.y_scaler.inverse_transform(pred_scaled)

        density = np.clip(pred[:, 0], D_MIN, D_MAX)
        tensile = np.maximum(pred[:, 1], 1e-4)
        er = np.maximum(pred[:, 2], 1e-4)
        efrf = er / tensile
        efrf = np.clip(efrf, 1e-4, 5.0)
        disintegration = np.maximum(pred[:, 3], 0.5)
        dissolution_tau = np.maximum(pred[:, 4], 1.0)

        penalty = np.zeros(n)
        penalty += np.where(tensile < TENSILE_MIN, (TENSILE_MIN - tensile)**2, 0.0)
        penalty += np.where(efrf >= 0.40, (efrf - 0.40)**2, 0.0)
        penalty += np.where(disintegration > 15.0, (disintegration - 15.0)**2, 0.0)
        penalty += np.where(dissolution_tau > 20.0, (dissolution_tau - 20.0)**2, 0.0)
        mcc_val = repaired[:, 1]
        penalty += np.where(mcc_val > self.bounds[1,1], (mcc_val - self.bounds[1,1])**2, 0.0)

        objectives = np.zeros((n, 3))
        objectives[:, 0] = -(repaired[:, 0]) + 100.0 * penalty
        objectives[:, 1] = efrf + 100.0 * penalty
        objectives[:, 2] = -density + 100.0 * penalty

        return objectives, None, repaired

    def _non_dominated_sort(self, objectives, violation):
        n = objectives.shape[0]
        fronts = []
        remaining = list(range(n))
        while remaining:
            front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if (objectives[j,0] <= objectives[i,0] and 
                        objectives[j,1] <= objectives[i,1] and 
                        objectives[j,2] <= objectives[i,2]) and \
                       (objectives[j,0] < objectives[i,0] or 
                        objectives[j,1] < objectives[i,1] or 
                        objectives[j,2] < objectives[i,2]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            fronts.append(front)
            remaining = [idx for idx in remaining if idx not in front]
        return fronts

    def _crowding_distance(self, objectives, front):
        if len(front) <= 2:
            return np.ones(len(front)) * np.inf
        dist = np.zeros(len(front))
        for obj_idx in range(objectives.shape[1]):
            sorted_idx = sorted(front, key=lambda i: objectives[i, obj_idx])
            dist[0] = np.inf
            dist[-1] = np.inf
            f_min = objectives[sorted_idx[0], obj_idx]
            f_max = objectives[sorted_idx[-1], obj_idx]
            if f_max - f_min > 1e-10:
                for k in range(1, len(sorted_idx)-1):
                    dist[k] += (objectives[sorted_idx[k+1], obj_idx] -
                                objectives[sorted_idx[k-1], obj_idx]) / (f_max - f_min)
        return dist

    def _crossover(self, p1, p2, eta=40):
        child1 = np.zeros(14)
        child2 = np.zeros(14)
        for i in range(14):
            u = np.random.random()
            if u <= 0.5:
                beta = (2*u) ** (1/(eta+1))
            else:
                beta = (1/(2*(1-u))) ** (1/(eta+1))
            child1[i] = 0.5 * ((1+beta)*p1[i] + (1-beta)*p2[i])
            child2[i] = 0.5 * ((1-beta)*p1[i] + (1+beta)*p2[i])
        return child1, child2

    def _mutate(self, child, eta=20, pm=1.0/14.0):
        for i in range(14):
            if np.random.random() < pm:
                u = np.random.random()
                if u <= 0.5:
                    delta = (2*u) ** (1/(eta+1)) - 1
                else:
                    delta = 1 - (2*(1-u)) ** (1/(eta+1))
                child[i] = child[i] + delta * (self.bounds[i,1] - self.bounds[i,0])
                child[i] = np.clip(child[i], self.bounds[i,0], self.bounds[i,1])
        return child

    def _tournament(self, pop, objectives, fronts, violation):
        idx1 = np.random.randint(0, len(pop))
        idx2 = np.random.randint(0, len(pop))
        rank1 = next((f for f, front in enumerate(fronts) if idx1 in front), len(fronts))
        rank2 = next((f for f, front in enumerate(fronts) if idx2 in front), len(fronts))
        if rank1 < rank2:
            return pop[idx1]
        elif rank2 < rank1:
            return pop[idx2]
        else:
            front = fronts[rank1]
            dist = self._crowding_distance(objectives, front)
            d1 = dist[front.index(idx1)]
            d2 = dist[front.index(idx2)]
            return pop[idx1] if d1 > d2 else pop[idx2]

    def run(self):
        rng = np.random.default_rng()
        pop = []
        for i in range(self.pop_size):
            if i < 0.3 * self.pop_size:
                api = rng.uniform(90, 95)
                mcc = rng.uniform(2.5, 4.0)
                binder = rng.uniform(3.5, 5.0)
                pvpp = rng.uniform(2, 4)
                mgst = rng.uniform(0.4, 0.8)
                moisture = rng.uniform(1.0, 3.0)
            else:
                api = rng.uniform(SLIDER_API_MIN, SLIDER_API_MAX)
                mcc = rng.uniform(BOUND_MCC_MIN, BOUND_MCC_MAX)
                binder = rng.uniform(BOUND_BINDER_MIN, BOUND_BINDER_MAX)
                pvpp = rng.uniform(BOUND_PVPP_MIN, BOUND_PVPP_MAX)
                mgst = rng.uniform(BOUND_MGST_MIN, BOUND_MGST_MAX)
                moisture = rng.uniform(SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX)
            pressure = rng.uniform(BOUND_PRESSURE_MIN, BOUND_PRESSURE_MAX)
            speed = rng.uniform(BOUND_SPEED_MIN, BOUND_SPEED_MAX)
            granule = rng.uniform(BOUND_GRANULE_MIN, BOUND_GRANULE_MAX)
            particle_size = rng.uniform(SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX)
            binder_grade = rng.integers(0, len(BINDER_GRADES))
            dwell_time = rng.uniform(SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX)
            friction = rng.uniform(SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX)
            decompression_time = rng.uniform(SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX)
            ind = np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule,
                            particle_size, moisture, binder_grade, dwell_time, friction, decompression_time])
            pop.append(self._repair(ind))
        pop = np.array(pop)

        for gen in range(self.generations):
            objectives, violation, pop = self._evaluate(pop)
            fronts = self._non_dominated_sort(objectives, violation)
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._tournament(pop, objectives, fronts, violation)
                p2 = self._tournament(pop, objectives, fronts, violation)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                offspring.append(self._repair(c1))
                if len(offspring) < self.pop_size:
                    offspring.append(self._repair(c2))
            offspring = np.array(offspring[:self.pop_size])
            combined = np.vstack([pop, offspring])
            obj_comb, viol_comb, _ = self._evaluate(combined)
            fronts_comb = self._non_dominated_sort(obj_comb, viol_comb)
            new_pop = []
            remaining = self.pop_size
            for front in fronts_comb:
                if len(front) <= remaining:
                    new_pop.extend(combined[front])
                    remaining -= len(front)
                else:
                    dist = self._crowding_distance(obj_comb, front)
                    sorted_idx = sorted(front, key=lambda i: dist[front.index(i)], reverse=True)
                    new_pop.extend(combined[sorted_idx[:remaining]])
                    remaining = 0
                    break
            pop = np.array(new_pop)

        objectives, violation, pop = self._evaluate(pop)
        fronts = self._non_dominated_sort(objectives, violation)
        return pop, objectives, fronts

# ================================================================
# PREDICTION
# ================================================================

def predict_pinn(model, scaler, y_scaler, inputs):
    if model is None:
        return 0.72, 2.0, 0.5, 0.25, 10.0, 10.0, 1.0
    try:
        aug = build_prediction_features(np.array([inputs]))
        scaled = scaler.transform(aug)
        X_t = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model.predict(X_t)[0]
            pred = y_scaler.inverse_transform([pred_scaled])[0]
        density = np.clip(pred[0], D_MIN, D_MAX)
        tensile = max(pred[1], 1e-4)
        er = max(pred[2], 1e-4)
        efrf = er / tensile
        disintegration = max(pred[3], 0.5)
        dissolution_tau = max(pred[4], 1.0)
        dissolution_beta = max(pred[5], 0.5)
        return density, tensile, er, efrf, disintegration, dissolution_tau, dissolution_beta
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.72, 2.0, 0.5, 0.25, 10.0, 10.0, 1.0

# ================================================================
# TRAIN MODEL – WITH CACHING
# ================================================================

CACHE_DIR = tempfile.gettempdir()
CHECKPOINT_PATH = os.path.join(CACHE_DIR, 'hubryd_final_31_clean.pt')

@st.cache_resource
def load_or_train():
    # Try to load from cache
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            model = MultiTaskPINN(ckpt['input_dim'], hidden=HIDDEN_SIZE)
            model.load_state_dict(ckpt['model_state'])
            scaler = ckpt['scaler']
            y_scaler = ckpt['y_scaler']
            features = ckpt['features']
            df = ckpt['df']
            st.success("✅ Model loaded from cache!")
            return model, scaler, y_scaler, features, df
        except Exception as e:
            st.warning(f"Cache load failed: {e}. Retraining...")
            if os.path.exists(CHECKPOINT_PATH):
                os.remove(CHECKPOINT_PATH)

    st.caption("🔄 Training model (clean 31 features)...")
    df = generate_pinn_data_with_features(N_SAMPLES)
    features = ['API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%',
                'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm',
                'Particle_Size_µm', 'Moisture_%', 'Binder_Grade',
                'Dwell_Time_ms', 'Friction', 'Decompression_Time_ms']
    
    # Build features from raw data
    X_raw = df[features].values
    X_aug = build_prediction_features(X_raw)
    n_features = X_aug.shape[1]
    st.info(f"✓ Number of features: {n_features} (expected 31)")

    y = df[['Density','Tensile_Strength_MPa','Elastic_Recovery_%',
            'Disintegration_Time_min','Dissolution_Tau','Dissolution_Beta']].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.caption(f"🖥️ Using device: {device}")
    model = MultiTaskPINN(n_features, hidden=HIDDEN_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    best_r2_tensile = -np.inf
    patience_counter = 0
    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(ADAM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = model.compute_loss(X_train_t, None, y_train_t, y_scaler, epoch, ADAM_EPOCHS)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred_scaled = model.predict(X_val_t)
            val_pred = y_scaler.inverse_transform(val_pred_scaled)
            val_true = y_scaler.inverse_transform(y_val_t.cpu().numpy())
            r2_tensile = r2_score(val_true[:, 1], val_pred[:, 1])
            r2_density = r2_score(val_true[:, 0], val_pred[:, 0])

        if epoch % 50 == 0:
            status_text.text(f"Epoch {epoch+1}/{ADAM_EPOCHS} - R² Tensile: {r2_tensile:.4f} | R² Density: {r2_density:.4f}")

        if r2_tensile > best_r2_tensile:
            best_r2_tensile = r2_tensile
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                st.info(f"Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break

        progress_bar.progress((epoch+1)/ADAM_EPOCHS)

    with torch.no_grad():
        test_pred_scaled = model.predict(torch.tensor(scaler.transform(build_prediction_features(X_test)), dtype=torch.float32))
        test_pred = y_scaler.inverse_transform(test_pred_scaled)
        test_true = y_scaler.inverse_transform(y_test)
        final_r2_tensile = r2_score(test_true[:, 1], test_pred[:, 1])
        final_r2_density = r2_score(test_true[:, 0], test_pred[:, 0])
    st.success(f"✅ Final R² Tensile: {final_r2_tensile:.4f} | Density: {final_r2_density:.4f}")

    checkpoint = {
        'model_state': model.state_dict(),
        'scaler': scaler,
        'y_scaler': y_scaler,
        'features': features,
        'df': df,
        'input_dim': n_features
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    st.success("✅ Model cached successfully!")

    return model, scaler, y_scaler, features, df

# ================================================================
# [REST OF CODE: run_model_comparison, generate_feasible_points,
#  plotting functions, PDF report, MAIN UI - similar to previous]
# ================================================================

# NOTE: For brevity, the plotting functions (plot_pareto_clean, 
# plot_sensitivity_bars, plot_dissolution_profile, 
# generate_enhanced_pdf_report) remain the same as in previous versions.
# The main UI also remains the same.

# ================================================================
# MAIN UI (simplified for display)
# ================================================================

st.markdown("""
<div style="background: #0b1a33; padding:1rem; border-radius:0.5rem; text-align:center; margin-bottom:1rem;">
    <h2 style="color:#fff; margin:0;">🧬 Hybrid AI For Multi‑Objective Tablet Optimization</h2>
    <p style="color:#64ffda; margin:0; font-size:1rem;">v29.27-R31</p>
    <p style="color:#aabbcc; margin:0; font-size:0.85rem;">Nile Valley University, Sudan</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📚 Physics Constraints")
    st.markdown(f"""
    ✅ **API:** {SLIDER_API_MIN:.0f}–{SLIDER_API_MAX:.0f}%  
    ✅ **Density:** 0.72–0.99  
    ✅ **Tensile:** ≥ {TENSILE_MIN:.2f} MPa  
    ✅ **EFRF:** &lt; 0.40 (feasible)  
    ✅ **Disintegration:** ≤ 15 min (USP)  
    ✅ **MCC:** {SLIDER_MCC_MIN:.1f}–{SLIDER_MCC_MAX:.1f}%  
    ✅ **PVPP:** {SLIDER_PVPP_MIN:.1f}–{SLIDER_PVPP_MAX:.1f}%  
    ✅ **MgSt:** {SLIDER_MGST_MIN:.2f}–{SLIDER_MGST_MAX:.2f}%  
    ✅ **Binder:** {SLIDER_BINDER_MIN:.1f}–{SLIDER_BINDER_MAX:.1f}%  
    ✅ **Moisture:** {SLIDER_MOISTURE_MIN:.1f}–{SLIDER_MOISTURE_MAX:.1f}%  
    ✅ **Pressure:** {BOUND_PRESSURE_MIN:.0f}–{BOUND_PRESSURE_MAX:.0f} MPa  
    ✅ **Speed:** {BOUND_SPEED_MIN:.0f}–{BOUND_SPEED_MAX:.0f} RPM  
    ✅ **NSGA‑II:** Pop=80, Gen=50 (3 objectives)
    """)
    st.caption("🔬 v29.27-R31 — CLEAN (31 features, cached)")

# Load model
try:
    model, scaler, y_scaler, features, df = load_or_train()
except Exception as e:
    st.error(f"❌ Training failed: {e}. Using dummy model.")
    model = None

# ... rest of UI (sliders, predict button, results display) ...

st.caption("📧 Contact: babuker@protonmail.com | 🏛️ Nile Valley University, Sudan")
