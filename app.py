"""
Hubryd AI – v29.28‑R32 (Adaptive Loss + Vectorized NSGA‑II)
Hybrid AI For Multi‑Objective Tablet Optimization
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
# CONSTANTS (ENHANCED)
# ================================================================
D_MIN = 0.75                         # NEW: minimum density raised
D_MAX = 0.99
TENSILE_MIN = 1.50
EFRF_MAX = 0.50
DISINTEGRATION_MAX = 15.0

# Slider ranges
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

# NSGA‑II bounds
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

# Training parameters (enhanced)
N_SAMPLES = 25000
ADAM_EPOCHS = 1200
PATIENCE = 100
NSGA_POP = 120
NSGA_GENS = 80
HIDDEN_SIZE = 512

# Loss weights (base values for adaptive loss)
W_DENSITY = 2.0
W_TENSILE = 500.0
W_ER = 5.0
W_PHYSICS = 0.5
W_EFRF_PENALTY = 50.0
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
# HELPER FUNCTIONS (unchanged)
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

def add_interaction_features(X_raw):
    pressure = X_raw[:, 5:6]
    binder = X_raw[:, 4:5]
    api = X_raw[:, 0:1]
    speed = X_raw[:, 6:7]
    mcc = X_raw[:, 1:2]
    pvpp = X_raw[:, 2:3]
    mgst = X_raw[:, 3:4]
    particle_size = X_raw[:, 8:9]
    moisture = X_raw[:, 9:10]
    binder_grade = X_raw[:, 10:11]
    dwell_time = X_raw[:, 11:12]
    friction = X_raw[:, 12:13]
    decompression_time = X_raw[:, 13:14]

    pressure_speed = np.clip(pressure / (speed + 0.1), 0, 1000)
    api_mcc = np.clip(api / (mcc + 0.1), 0, 1000)
    binder_speed = np.clip(binder / (speed + 0.1), 0, 100)
    pressure_binder = pressure * binder
    pressure_api = pressure * api
    api_pvpp = api * pvpp
    binder_mgst = binder * mgst
    mcc_pvpp = mcc * pvpp
    api2 = api ** 2
    pressure2 = pressure ** 2
    binder2 = binder ** 2
    speed2 = speed ** 2

    # Extra interactions
    binder_pvpp = binder * pvpp
    mcc_moisture = mcc * moisture
    api_binder = api * binder
    pressure_moisture = pressure * moisture
    speed_moisture = speed * moisture
    granule_particle = particle_size * pressure
    dwell_speed = dwell_time * speed
    friction_pressure = friction * pressure

    particle_pressure = particle_size * pressure
    moisture_pressure = moisture * pressure
    particle_moisture = particle_size * moisture
    dwell_pressure = dwell_time * pressure
    friction_pressure_old = friction * pressure

    return np.concatenate([
        X_raw,
        pressure_binder, pressure_api,
        pressure_speed, api_mcc, binder_speed,
        api_pvpp, binder_mgst, mcc_pvpp,
        binder_pvpp, mcc_moisture, api_binder,
        pressure_moisture, speed_moisture,
        granule_particle, dwell_speed, friction_pressure,
        api2, pressure2, binder2, speed2,
        particle_pressure, moisture_pressure,
        particle_moisture, dwell_pressure, friction_pressure_old
    ], axis=1)

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
# DATA GENERATION (same as v29.27 – physically stable)
# ================================================================

def generate_pinn_data(n_samples=N_SAMPLES, random_state=42):
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

    X = np.column_stack([
        api_n, mcc_n, pvpp_n, mgst_n, binder_n,
        pressure_raw, speed_raw, granule_raw,
        particle_size_raw, moisture_n, binder_grade_raw,
        dwell_time_raw, friction_raw, decompression_time_raw
    ])

    # ----- Density (improved physics) -----
    k_heckel = 0.025 + 0.0001 * pressure_raw
    A_heckel = 1.0 + 0.01 * (api_n - 85.0) - 0.05 * binder_n
    x_val = k_heckel * pressure_raw + A_heckel
    D_heckel = 1.0 - np.exp(-x_val)
    D_heckel = np.clip(D_heckel, 0.75, 0.99)

    a_kawakita = 0.80 + 0.04 * (mcc_n - 1.5) / 6.5 + 0.02 * (binder_n - 1.4) / 4.6
    a_kawakita = np.clip(a_kawakita, 0.78, 0.92)
    b_kawakita = 0.002 + 0.001 * (binder_n - 1.4) / 4.6 + 0.0005 * (mcc_n - 1.5) / 6.5
    b_kawakita = np.clip(b_kawakita, 0.0005, 0.005)
    b_kawakita *= (1.0 + rng.normal(0, 0.03, n_samples))
    b_kawakita = np.clip(b_kawakita, 0.0003, 0.006)
    D_kawakita = 1.0 - pressure_raw / (a_kawakita * pressure_raw + 1.0 / b_kawakita)
    D_kawakita = np.clip(D_kawakita, 0.75, 0.99)

    moisture_effect = -0.002 * (moisture_n - 2.0)
    moisture_effect = np.clip(moisture_effect, -0.02, 0.01)
    particle_effect = -0.003 * (particle_size_raw - 50) / 150
    particle_effect = np.clip(particle_effect, -0.02, 0.01)
    speed_effect = -0.002 * (speed_raw - 15) / 15
    speed_effect = np.clip(speed_effect, -0.015, 0.0)
    mgst_effect = -0.01 * (mgst_n - 0.2)
    mgst_effect = np.clip(mgst_effect, -0.02, 0.005)

    pressure_norm = (pressure_raw - SLIDER_PRESSURE_MIN) / (SLIDER_PRESSURE_MAX - SLIDER_PRESSURE_MIN)
    mcc_weight = 0.3 + 0.4 * (mcc_n - 1.5) / 6.5
    mcc_weight = np.clip(mcc_weight, 0.3, 0.7)
    w_heckel = pressure_norm * (1.0 - mcc_weight) + mcc_weight * 0.7
    w_heckel = np.clip(w_heckel, 0.2, 0.9)
    w_kawakita = 1.0 - w_heckel

    D = w_heckel * D_heckel + w_kawakita * D_kawakita
    D += moisture_effect + particle_effect + speed_effect + mgst_effect
    D += rng.normal(0, 0.003, n_samples)
    D = np.clip(D, 0.75, 0.99)

    # Tensile
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
    strength *= rng.normal(1.0, 0.02, n_samples)
    strength = np.clip(strength, 0.5, 6.0)

    # Elastic Recovery
    er_base = (1.8 + 0.3 * (api_n - 85.0)/10.0 +
               0.08 * (speed_raw - 10.0)/30.0 -
               0.1 * (pressure_raw - 100.0)/150.0 +
               0.02 * (decompression_time_raw - 35.0)/30.0)
    er_base = er_base * (1.0 - 0.15 * (D - 0.4))
    er = er_base + rng.normal(0, 0.02, n_samples)
    er = np.clip(er, 0.5, 4.0)

    # Disintegration & dissolution
    disintegration = predict_disintegration_time(strength, pvpp_n, api_n, binder_n, moisture_n)
    disintegration += rng.normal(0, 0.15, n_samples)
    disintegration = np.clip(disintegration, 1.0, 30.0)

    dissolution_params = predict_dissolution_profile(api_n, pvpp_n, particle_size_raw, disintegration)
    dissolution_tau = dissolution_params['tau'] + rng.normal(0, 0.15, n_samples)
    dissolution_tau = np.clip(dissolution_tau, 2.0, 20.0)
    dissolution_beta = np.clip(dissolution_params['beta'], 0.8, 2.5)

    feature_names = [
        'API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%',
        'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm',
        'Particle_Size_µm', 'Moisture_%', 'Binder_Grade',
        'Dwell_Time_ms', 'Friction', 'Decompression_Time_ms'
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df['Density'] = D
    df['Tensile_Strength_MPa'] = strength
    df['Elastic_Recovery_%'] = er
    df['Disintegration_Time_min'] = disintegration
    df['Dissolution_Tau'] = dissolution_tau
    df['Dissolution_Beta'] = dissolution_beta

    return df, feature_names

# ================================================================
# PINN MODEL – with ADAPTIVE LOSS
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
        self.res3 = ResidualBlock(hidden, dropout=0.05)
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
        x = self.res3(x)
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

    # ================================================================
    # ADAPTIVE LOSS (NEW) – replaces the old compute_loss
    # ================================================================
    def compute_loss(self, X_scaled, X_raw, y_true, y_scaler, epoch=0, total_epochs=ADAM_EPOCHS):
        pressure = X_raw[:, 5].view(-1, 1)
        mcc = X_raw[:, 1].view(-1, 1)
        api = X_raw[:, 0].view(-1, 1)
        binder = X_raw[:, 4].view(-1, 1)
        pvpp = X_raw[:, 2].view(-1, 1)
        moisture = X_raw[:, 9].view(-1, 1)

        y_pred = self.forward(X_scaled)
        density_pred = y_pred[:, 0:1]
        tensile_pred = y_pred[:, 1:2]
        er_pred = y_pred[:, 2:3]
        k_heckel_pred = y_pred[:, 3:4]
        A_heckel_pred = y_pred[:, 4:5]
        a_kawakita_pred = y_pred[:, 5:6]
        b_kawakita_pred = y_pred[:, 6:7]
        disintegration_pred = y_pred[:, 7:8]
        dissolution_tau_pred = y_pred[:, 8:9]
        dissolution_beta_pred = y_pred[:, 9:10]

        # --- Data loss ---
        mse = nn.MSELoss()
        data_loss = (
            W_DENSITY * mse(density_pred, y_true[:, 0:1]) +
            W_TENSILE * mse(tensile_pred, y_true[:, 1:2]) +
            W_ER * mse(er_pred, y_true[:, 2:3]) +
            W_DISINTEGRATION * mse(disintegration_pred, y_true[:, 3:4]) +
            W_DISSOLUTION * (mse(dissolution_tau_pred, y_true[:, 4:5]) +
                             mse(dissolution_beta_pred, y_true[:, 5:6]))
        )

        # --- Physics loss ---
        scale_dens, mean_dens = y_scaler.scale_[0], y_scaler.mean_[0]
        scale_tensile, mean_tensile = y_scaler.scale_[1], y_scaler.mean_[1]
        scale_er, mean_er = y_scaler.scale_[2], y_scaler.mean_[2]

        density_real = density_pred * scale_dens + mean_dens
        tensile_real = tensile_pred * scale_tensile + mean_tensile
        er_real = er_pred * scale_er + mean_er

        # 1) Heckel
        heckel_lhs = torch.log(1.0 / torch.clamp(1.0 - density_real, min=1e-4))
        heckel_rhs = k_heckel_pred * pressure + A_heckel_pred
        heckel_loss = mse(heckel_lhs, heckel_rhs)

        # 2) Kawakita
        epsilon = torch.clamp(1.0 - density_real, min=1e-4)
        kawakita_lhs = pressure / epsilon
        kawakita_rhs = a_kawakita_pred * pressure + 1.0 / b_kawakita_pred
        kawakita_loss = mse(kawakita_lhs, kawakita_rhs)

        # 3) EFRF
        efrf_real = er_real / torch.clamp(tensile_real, min=1e-4)
        efrf_penalty = torch.mean(torch.relu(efrf_real - 0.50) ** 2) * W_EFRF_PENALTY

        # 4) Disintegration
        disin_physics = (2.0 + 0.5 * tensile_real -
                         5.0 * torch.exp(-0.5 * pvpp) +
                         0.1 * (api - 80) +
                         0.2 * (binder - 2.0) -
                         0.1 * moisture)
        disin_physics = torch.clamp(disin_physics, 1.0, 30.0)
        physics_disin_loss = mse(disintegration_pred, disin_physics)

        # 5) Dissolution
        tau_physics = 5.0 + 0.5 * disintegration_pred - 0.1 * pvpp + 0.05 * (api - 80)
        tau_physics = torch.clamp(tau_physics, 2.0, 20.0)
        physics_tau_loss = mse(dissolution_tau_pred, tau_physics)

        physics_loss = W_PHYSICS * (heckel_loss + kawakita_loss + efrf_penalty +
                                    physics_disin_loss + physics_tau_loss)

        # --- Adaptive weighting ---
        adaptive_weight = torch.exp(-epoch / (0.25 * total_epochs))
        total_loss = adaptive_weight * data_loss + (1 - adaptive_weight) * physics_loss

        return total_loss

# ================================================================
# NSGA-II – VECTORIZED EVALUATION (NEW)
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

    # ================================================================
    # VECTORIZED _evaluate (NEW) – batch processing
    # ================================================================
    def _evaluate(self, population):
        repaired = self._repair_batch(population)
        aug = add_interaction_features(repaired)
        scaled = self.scaler.transform(aug)
        X_t = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = self.model.predict(X_t)
            pred = self.y_scaler.inverse_transform(pred_scaled)

        density = np.clip(pred[:, 0], 0.75, D_MAX)
        tensile = np.maximum(pred[:, 1], 1e-4)
        er = np.maximum(pred[:, 2], 1e-4)
        efrf = np.clip(er / tensile, 1e-4, 5.0)
        disintegration = np.maximum(pred[:, 3], 0.5)
        dissolution_tau = np.maximum(pred[:, 4], 1.0)

        # --- Penalties (fully vectorized) ---
        penalty = (
            np.square(np.maximum(TENSILE_MIN - tensile, 0)) +
            np.square(np.maximum(efrf - 0.40, 0)) +
            np.square(np.maximum(disintegration - 15.0, 0)) +
            np.square(np.maximum(dissolution_tau - 20.0, 0)) +
            np.square(np.maximum(repaired[:, 1] - self.bounds[1, 1], 0)) +
            np.square(np.maximum(0.80 - density, 0)) * 50.0
        )

        # --- Objectives (3) ---
        objectives = np.column_stack([
            -repaired[:, 0] + 100.0 * penalty,   # maximize API
            efrf + 100.0 * penalty,              # minimize EFRF
            -density + 100.0 * penalty           # maximize density
        ])

        return objectives, None, repaired

    # ------------------------------------------------------------------
    # The rest of NSGAII (non_dominated_sort, crowding_distance, etc.)
    # remains unchanged – they work with any number of objectives.
    # ------------------------------------------------------------------

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
# PREDICTION AND PLOTTING (minor UI updates)
# ================================================================

def predict_pinn(model, scaler, y_scaler, inputs):
    if model is None:
        return 0.75, 2.0, 0.5, 0.25, 10.0, 10.0, 1.0
    try:
        aug = add_interaction_features(np.array([inputs]))[0]
        scaled = scaler.transform([aug])
        X_t = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model.predict(X_t)[0]
            pred = y_scaler.inverse_transform([pred_scaled])[0]
        density = np.clip(pred[0], 0.75, D_MAX)
        tensile = max(pred[1], 1e-4)
        er = max(pred[2], 1e-4)
        efrf = er / tensile
        disintegration = max(pred[3], 0.5)
        dissolution_tau = max(pred[4], 1.0)
        dissolution_beta = max(pred[5], 0.5)
        return density, tensile, er, efrf, disintegration, dissolution_tau, dissolution_beta
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.75, 2.0, 0.5, 0.25, 10.0, 10.0, 1.0

def plot_pareto_clean(objectives, fronts, balanced_solution=None, feasible_df=None,
                      tested_point=None, efrf_max=0.40):
    if fronts is None or len(fronts) == 0 or len(fronts[0]) == 0:
        return None
    front = fronts[0]
    try:
        api_vals = -objectives[front, 0]
        efrf_vals = objectives[front, 1]
    except Exception:
        return None

    df_front = pd.DataFrame({'API': api_
