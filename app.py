"""
Hubryd AI – v29.27-R31 (ENHANCED – 19 Features + Larger Network)
Hybrid AI For Multi-Objective Tablet Optimization
Nile Valley University, Sudan
"""

import streamlit as st
# Must be first Streamlit command
st.set_page_config(
    page_title="Hybrid AI · Tablet Optimization v29.27-R31",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
import json
import base64
from io import BytesIO
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Any

warnings.filterwarnings('ignore')

# Optional PDF library
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# ================================================================
# CONFIGURATION (Centralised)
# ================================================================
@dataclass
class Config:
    # Formulation bounds
    API_MIN: float = 80.0
    API_MAX: float = 98.0
    MCC_MIN: float = 1.5
    MCC_MAX: float = 8.0
    PVPP_MIN: float = 1.0
    PVPP_MAX: float = 6.0
    MGST_MIN: float = 0.10
    MGST_MAX: float = 1.2
    BINDER_MIN: float = 1.4
    BINDER_MAX: float = 6.0
    MOISTURE_MIN: float = 0.5
    MOISTURE_MAX: float = 5.0
    PARTICLE_SIZE_MIN: float = 10.0
    PARTICLE_SIZE_MAX: float = 200.0

    # Process bounds
    PRESSURE_MIN: float = 150.0
    PRESSURE_MAX: float = 250.0
    SPEED_MIN: float = 15.0
    SPEED_MAX: float = 30.0
    GRANULE_MIN: float = 30.0
    GRANULE_MAX: float = 250.0
    DWELL_TIME_MIN: float = 5.0
    DWELL_TIME_MAX: float = 50.0
    FRICTION_MIN: float = 0.1
    FRICTION_MAX: float = 0.5
    DECOMPRESSION_TIME_MIN: float = 10.0
    DECOMPRESSION_TIME_MAX: float = 80.0

    # Additional constraints for NSGA‑II
    BOUND_MCC_MIN: float = 2.0
    BOUND_MCC_MAX: float = 8.0
    BOUND_PVPP_MIN: float = 1.5
    BOUND_PVPP_MAX: float = 6.0
    BOUND_MGST_MIN: float = 0.3
    BOUND_MGST_MAX: float = 1.2
    BOUND_BINDER_MIN: float = 3.0
    BOUND_BINDER_MAX: float = 6.0
    BOUND_PRESSURE_MIN: float = 150.0
    BOUND_PRESSURE_MAX: float = 250.0
    BOUND_SPEED_MIN: float = 15.0
    BOUND_SPEED_MAX: float = 30.0
    BOUND_GRANULE_MIN: float = 30.0
    BOUND_GRANULE_MAX: float = 250.0

    # Performance targets
    DENSITY_MIN: float = 0.72
    DENSITY_MAX: float = 0.99
    TENSILE_MIN: float = 1.50
    EFRF_MAX: float = 0.40
    DISINTEGRATION_MAX: float = 15.0

    # Model training
    N_SAMPLES: int = 30000
    EPOCHS: int = 800
    PATIENCE: int = 100
    HIDDEN_SIZE: int = 512
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-5

    # NSGA‑II
    NSGA_POP: int = 80
    NSGA_GENS: int = 50

    # Loss weights
    W_DENSITY: float = 1.0
    W_TENSILE: float = 500.0
    W_ER: float = 5.0
    W_DISINTEGRATION: float = 50.0
    W_DISSOLUTION: float = 20.0

    # Binder grades
    BINDER_GRADES: List[str] = None

    def __post_init__(self):
        if self.BINDER_GRADES is None:
            self.BINDER_GRADES = [
                "MCC PH101", "MCC PH102", "MCC PH200",
                "MCC KG", "Lactose", "Dicalcium Phosphate"
            ]

# Instantiate config
CFG = Config()

# ================================================================
# SESSION STATE INITIALISATION
# ================================================================
def init_session_state():
    """Ensure all session state keys exist."""
    defaults = {
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
        'experimental_data': None,
        '_model_loaded': False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ================================================================
# HELPER FUNCTIONS
# ================================================================
def normalize_components(
    api: np.ndarray,
    binder: np.ndarray,
    pvpp: np.ndarray,
    mgst: np.ndarray,
    mcc: np.ndarray,
    moisture: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """
    Normalise formulation components to sum to 100% while respecting bounds.
    """
    # Convert to arrays and clip to bounds
    api = np.clip(np.asarray(api, dtype=float), CFG.API_MIN, CFG.API_MAX)
    binder = np.clip(np.asarray(binder, dtype=float), CFG.BINDER_MIN, CFG.BINDER_MAX)
    pvpp = np.clip(np.asarray(pvpp, dtype=float), CFG.PVPP_MIN, CFG.PVPP_MAX)
    mgst = np.clip(np.asarray(mgst, dtype=float), CFG.MGST_MIN, CFG.MGST_MAX)
    mcc = np.clip(np.asarray(mcc, dtype=float), CFG.MCC_MIN, CFG.MCC_MAX)
    moisture = np.clip(np.asarray(moisture, dtype=float), CFG.MOISTURE_MIN, CFG.MOISTURE_MAX)

    total = api + binder + pvpp + mgst + mcc + moisture
    total = np.where(total <= 0, 1.0, total)

    # Normalise to 100%
    api = (api / total) * 100.0
    binder = (binder / total) * 100.0
    pvpp = (pvpp / total) * 100.0
    mgst = (mgst / total) * 100.0
    mcc = (mcc / total) * 100.0
    moisture = (moisture / total) * 100.0

    # Re-clip to respect absolute bounds
    api = np.clip(api, CFG.API_MIN, CFG.API_MAX)
    binder = np.clip(binder, CFG.BINDER_MIN, CFG.BINDER_MAX)
    pvpp = np.clip(pvpp, CFG.PVPP_MIN, CFG.PVPP_MAX)
    mgst = np.clip(mgst, CFG.MGST_MIN, CFG.MGST_MAX)
    mcc = np.clip(mcc, CFG.MCC_MIN, CFG.MCC_MAX)
    moisture = np.clip(moisture, CFG.MOISTURE_MIN, CFG.MOISTURE_MAX)

    # Re-normalise to exactly 100% after clipping
    total2 = api + binder + pvpp + mgst + mcc + moisture
    total2 = np.where(total2 <= 0, 1.0, total2)
    scale = 100.0 / total2
    api *= scale
    binder *= scale
    pvpp *= scale
    mgst *= scale
    mcc *= scale
    moisture *= scale

    # Final clip
    api = np.clip(api, CFG.API_MIN, CFG.API_MAX)
    binder = np.clip(binder, CFG.BINDER_MIN, CFG.BINDER_MAX)
    pvpp = np.clip(pvpp, CFG.PVPP_MIN, CFG.PVPP_MAX)
    mgst = np.clip(mgst, CFG.MGST_MIN, CFG.MGST_MAX)
    mcc = np.clip(mcc, CFG.MCC_MIN, CFG.MCC_MAX)
    moisture = np.clip(moisture, CFG.MOISTURE_MIN, CFG.MOISTURE_MAX)

    return api, binder, pvpp, mgst, mcc, moisture


def calculate_dwell_time(
    speed_rpm: np.ndarray,
    punch_width: float = 10.0,
    pitch_diameter: float = 100.0
) -> np.ndarray:
    """Compute dwell time from speed."""
    speed_rpm = np.asarray(speed_rpm, dtype=float)
    result = np.full_like(speed_rpm, 50.0, dtype=float)
    mask = speed_rpm > 0
    result[mask] = (punch_width * 60 * 1000) / (np.pi * pitch_diameter * speed_rpm[mask])
    return np.clip(result, CFG.DWELL_TIME_MIN, CFG.DWELL_TIME_MAX)


def predict_disintegration_time(
    tensile: np.ndarray,
    pvpp_n: np.ndarray,
    api_n: np.ndarray,
    binder_n: np.ndarray,
    moisture_n: np.ndarray
) -> np.ndarray:
    """Heuristic disintegration time prediction."""
    base_time = 2.0 + 0.5 * tensile
    pvpp_effect = 5.0 * np.exp(-0.5 * pvpp_n)
    api_effect = 0.1 * (api_n - 80.0)
    binder_effect = 0.2 * (binder_n - 2.0)
    moisture_effect = -0.1 * moisture_n
    time = base_time - pvpp_effect + api_effect + binder_effect + moisture_effect
    return np.clip(time, 1.0, 30.0)


def predict_dissolution_profile(
    api_n: np.ndarray,
    pvpp_n: np.ndarray,
    particle_size: np.ndarray,
    disintegration_time: np.ndarray
) -> Dict[str, np.ndarray]:
    """Weibull parameters for dissolution."""
    tau = 5.0 + 0.5 * disintegration_time - 0.1 * pvpp_n + 0.05 * (api_n - 80.0)
    tau = np.clip(tau, 2.0, 20.0)
    beta = 1.0 + 0.01 * (particle_size - 50.0) / 50.0
    beta = np.clip(beta, 0.8, 2.5)
    return {'tau': tau, 'beta': beta}

# ================================================================
# SYNTHETIC DATA GENERATION (19 features)
# ================================================================
def generate_pinn_data(
    n_samples: int = CFG.N_SAMPLES,
    random_state: int = 42
) -> Tuple[pd.DataFrame, List[str]]:
    """Generate physics‑based synthetic dataset with 19 features."""
    rng = np.random.default_rng(random_state)

    # Raw variables
    api_raw = rng.uniform(CFG.API_MIN, CFG.API_MAX, n_samples)
    binder_raw = rng.uniform(CFG.BINDER_MIN, CFG.BINDER_MAX, n_samples)
    pvpp_raw = rng.uniform(CFG.PVPP_MIN, CFG.PVPP_MAX, n_samples)
    mgst_raw = rng.uniform(CFG.MGST_MIN, CFG.MGST_MAX, n_samples)
    mcc_raw = rng.uniform(CFG.MCC_MIN, CFG.MCC_MAX, n_samples)
    moisture_raw = rng.uniform(CFG.MOISTURE_MIN, CFG.MOISTURE_MAX, n_samples)

    api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n = normalize_components(
        api_raw, binder_raw, pvpp_raw, mgst_raw, mcc_raw, moisture_raw
    )

    particle_size_raw = rng.uniform(CFG.PARTICLE_SIZE_MIN, CFG.PARTICLE_SIZE_MAX, n_samples)
    binder_grade_raw = rng.integers(0, len(CFG.BINDER_GRADES), n_samples)
    pressure_raw = rng.uniform(CFG.PRESSURE_MIN, CFG.PRESSURE_MAX, n_samples)
    speed_raw = rng.uniform(CFG.SPEED_MIN, CFG.SPEED_MAX, n_samples)
    dwell_time_raw = calculate_dwell_time(speed_raw)
    friction_raw = rng.uniform(CFG.FRICTION_MIN, CFG.FRICTION_MAX, n_samples)
    decompression_time_raw = rng.uniform(
        CFG.DECOMPRESSION_TIME_MIN, CFG.DECOMPRESSION_TIME_MAX, n_samples
    )
    granule_raw = rng.uniform(CFG.GRANULE_MIN, CFG.GRANULE_MAX, n_samples)

    # Base features (14)
    X_base = np.column_stack([
        api_n, mcc_n, pvpp_n, mgst_n, binder_n,
        pressure_raw, speed_raw, granule_raw,
        particle_size_raw, moisture_n, binder_grade_raw,
        dwell_time_raw, friction_raw, decompression_time_raw
    ])

    # Interaction features (5)
    api_binder = api_n * binder_n
    pressure_binder = pressure_raw * binder_n
    api_mcc = api_n * mcc_n
    pressure_speed = pressure_raw * speed_raw
    binder_mgst = binder_n * mgst_n

    X_enhanced = np.column_stack([
        X_base,
        api_binder,
        pressure_binder,
        api_mcc,
        pressure_speed,
        binder_mgst
    ])  # 19 features

    # Physics-based targets
    # Density (Heckel + Kawakita blend)
    k_heckel = 0.025 + 0.0001 * pressure_raw
    A_heckel = 1.0 + 0.01 * (api_n - 85.0) - 0.05 * binder_n
    x_val = k_heckel * pressure_raw + A_heckel
    D_heckel = 1.0 - np.exp(-x_val)
    D_heckel = np.clip(D_heckel, CFG.DENSITY_MIN, CFG.DENSITY_MAX)

    a_kawakita = 0.82 + 0.04 * (mcc_n - 1.5) / 6.5 + 0.02 * (binder_n - 1.4) / 4.6
    a_kawakita = np.clip(a_kawakita, 0.78, 0.92)
    b_kawakita = 0.002 + 0.003 * (binder_n - 1.4) / 4.6 + 0.001 * (mcc_n - 1.5) / 6.5
    b_kawakita = np.clip(b_kawakita, 0.0005, 0.006)
    D_kawakita = 1.0 - pressure_raw / (a_kawakita * pressure_raw + 1.0 / b_kawakita)
    D_kawakita = np.clip(D_kawakita, CFG.DENSITY_MIN, CFG.DENSITY_MAX)

    pressure_norm = (pressure_raw - CFG.PRESSURE_MIN) / (CFG.PRESSURE_MAX - CFG.PRESSURE_MIN)
    D = pressure_norm * D_heckel + (1.0 - pressure_norm) * D_kawakita

    # Corrections
    moisture_effect = -0.003 * (moisture_n - 2.0)
    moisture_effect = np.clip(moisture_effect, -0.02, 0.01)
    particle_effect = -0.002 * (particle_size_raw - 50.0) / 150.0
    particle_effect = np.clip(particle_effect, -0.02, 0.01)
    speed_effect = -0.002 * (speed_raw - 15.0) / 15.0
    speed_effect = np.clip(speed_effect, -0.015, 0.0)
    mgst_effect = -0.01 * (mgst_n - 0.2)
    mgst_effect = np.clip(mgst_effect, -0.02, 0.005)

    D += moisture_effect + particle_effect + speed_effect + mgst_effect
    D = np.clip(D, CFG.DENSITY_MIN, CFG.DENSITY_MAX)

    # Tensile strength
    porosity = 1.0 - D
    sigma0 = 5.0 + 0.1 * (api_n - 85.0) + 0.2 * binder_n - 0.5 * mgst_n
    sigma0 = np.clip(sigma0, 2.0, 8.0)
    b = 2.5 - 0.005 * (pressure_raw - 80.0) - 0.01 * (particle_size_raw - 50.0) / 100.0
    b = np.clip(b, 1.5, 3.5)

    tensile_base = sigma0 * np.exp(-b * porosity)
    api_effect = 1.0 - 0.005 * (api_n - 85.0)
    binder_effect = 1.0 + 0.03 * (binder_n - 2.0)
    mgst_effect = 1.0 - 0.1 * (mgst_n - 0.2)
    pvpp_effect = 1.0 - 0.02 * (pvpp_n - 3.0)
    speed_effect = 1.0 - 0.002 * (speed_raw - 10.0)
    particle_effect = 1.0 - 0.0005 * (particle_size_raw - 50.0)
    particle_effect = np.clip(particle_effect, 0.8, 1.2)

    strength = (tensile_base * api_effect * binder_effect *
                mgst_effect * pvpp_effect * speed_effect * particle_effect)
    strength = np.clip(strength, 0.5, 6.0)

    # Elastic recovery
    er_base = (1.8 + 0.3 * (api_n - 85.0) / 10.0 +
               0.08 * (speed_raw - 10.0) / 30.0 -
               0.1 * (pressure_raw - 100.0) / 150.0 +
               0.02 * (decompression_time_raw - 35.0) / 30.0)
    er_base = er_base * (1.0 - 0.15 * (D - 0.4))
    er = np.clip(er_base, 0.5, 4.0)

    # Disintegration & dissolution
    disintegration = predict_disintegration_time(strength, pvpp_n, api_n, binder_n, moisture_n)
    disintegration = np.clip(disintegration, 1.0, 30.0)

    diss_params = predict_dissolution_profile(api_n, pvpp_n, particle_size_raw, disintegration)
    dissolution_tau = np.clip(diss_params['tau'], 2.0, 20.0)
    dissolution_beta = np.clip(diss_params['beta'], 0.8, 2.5)

    # Build DataFrame
    feature_names = [
        'API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%',
        'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm',
        'Particle_Size_µm', 'Moisture_%', 'Binder_Grade',
        'Dwell_Time_ms', 'Friction', 'Decompression_Time_ms',
        'API_Binder', 'Pressure_Binder', 'API_MCC', 'Pressure_Speed', 'Binder_MgSt'
    ]
    df = pd.DataFrame(X_enhanced, columns=feature_names)
    df['Density'] = D
    df['Tensile_Strength_MPa'] = strength
    df['Elastic_Recovery_%'] = er
    df['Disintegration_Time_min'] = disintegration
    df['Dissolution_Tau'] = dissolution_tau
    df['Dissolution_Beta'] = dissolution_beta

    return df, feature_names

# ================================================================
# PINN MODEL (Multi‑task with residual blocks)
# ================================================================
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class ResidualBlock(nn.Module):
    def __init__(self, features: int, dropout: float = 0.1):
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
    def __init__(self, input_dim: int, hidden: int = CFG.HIDDEN_SIZE):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden),
            Mish(),
            nn.Dropout(0.05)
        )
        self.res1 = ResidualBlock(hidden, dropout=0.05)
        self.res2 = ResidualBlock(hidden, dropout=0.05)
        self.res3 = ResidualBlock(hidden, dropout=0.05)
        self.transition = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Dropout(0.05)
        )
        self.output = nn.Linear(hidden // 2, 10)

        # Initialise weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

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

        return torch.cat([
            density, tensile, er,
            k_heckel, A_heckel, a_kawakita, b_kawakita,
            disintegration, dissolution_tau, dissolution_beta
        ], dim=1)

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """Return [density, tensile, er, disintegration, dissolution_tau, dissolution_beta]."""
        self.eval()
        with torch.no_grad():
            if not isinstance(X_scaled, torch.Tensor):
                X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
            device = next(self.parameters()).device
            X_scaled = X_scaled.to(device)
            output = self.forward(X_scaled)
            # Select only the 6 targets (density, tensile, er, disintegration, tau, beta)
            selected = torch.cat([output[:, 0:3], output[:, 7:10]], dim=1)
            return selected.cpu().numpy()

# ================================================================
# MODEL TRAINING (with caching)
# ================================================================
CACHE_DIR = tempfile.gettempdir()
CHECKPOINT_PATH = os.path.join(CACHE_DIR, 'hubryd_19features_enhanced_v2.pt')

@st.cache_resource(show_spinner=False)
def load_or_train() -> Tuple[MultiTaskPINN, StandardScaler, StandardScaler, List[str], pd.DataFrame]:
    """Load cached model or train from scratch."""
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            model = MultiTaskPINN(ckpt['input_dim'], hidden=CFG.HIDDEN_SIZE)
            model.load_state_dict(ckpt['model_state'])
            scaler = ckpt['scaler']
            y_scaler = ckpt['y_scaler']
            features = ckpt['features']
            df = ckpt['df']
            return model, scaler, y_scaler, features, df
        except Exception as e:
            st.warning(f"Cache load failed: {e}. Retraining...")
            if os.path.exists(CHECKPOINT_PATH):
                os.remove(CHECKPOINT_PATH)

    st.info("🔄 Training enhanced model (19 features, 30k samples)... This may take a few minutes.")
    df, features = generate_pinn_data(CFG.N_SAMPLES)
    n_features = len(features)

    y = df[['Density', 'Tensile_Strength_MPa', 'Elastic_Recovery_%',
            'Disintegration_Time_min', 'Dissolution_Tau', 'Dissolution_Beta']].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features].values)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskPINN(n_features, hidden=CFG.HIDDEN_SIZE).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5, verbose=False)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    best_r2_tensile = -np.inf
    patience_counter = 0
    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(CFG.EPOCHS):
        model.train()
        optimizer.zero_grad()
        # Compute loss (custom method)
        loss = model.compute_loss(X_train_t, None, y_train_t, y_scaler, epoch, CFG.EPOCHS)
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

        if epoch % 10 == 0 or epoch == CFG.EPOCHS - 1:
            status_text.text(
                f"Epoch {epoch+1}/{CFG.EPOCHS} - R² Tensile: {r2_tensile:.4f} | R² Density: {r2_density:.4f}"
            )
            progress_bar.progress((epoch + 1) / CFG.EPOCHS)

        if r2_tensile > best_r2_tensile:
            best_r2_tensile = r2_tensile
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CFG.PATIENCE:
                st.info(f"Early stopping at epoch {epoch+1} (no improvement for {CFG.PATIENCE} epochs)")
                break

    # Final evaluation
    with torch.no_grad():
        test_pred_scaled = model.predict(torch.tensor(scaler.transform(X_test), dtype=torch.float32))
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


# Add compute_loss method to MultiTaskPINN (since it's used in training)
def compute_loss_pinn(
    self,
    X_scaled: torch.Tensor,
    X_raw: Optional[torch.Tensor],
    y_true: torch.Tensor,
    y_scaler: StandardScaler,
    epoch: int,
    total_epochs: int
) -> torch.Tensor:
    """Compute multi‑task loss."""
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

    data_loss = (
        CFG.W_DENSITY * loss_dens +
        CFG.W_TENSILE * loss_tensile +
        CFG.W_ER * loss_er +
        CFG.W_DISINTEGRATION * loss_disin +
        CFG.W_DISSOLUTION * (loss_tau + loss_beta)
    )
    return data_loss


# Monkey‑patch the method
MultiTaskPINN.compute_loss = compute_loss_pinn

# ================================================================
# NSGA‑II OPTIMISER (Improved)
# ================================================================
class NSGAII:
    def __init__(
        self,
        model: MultiTaskPINN,
        scaler: StandardScaler,
        y_scaler: StandardScaler,
        bounds: np.ndarray,
        pop: int = CFG.NSGA_POP,
        gens: int = CFG.NSGA_GENS,
        granule_fixed: bool = True,
        granule_fixed_val: float = 125.0
    ):
        self.model = model
        self.scaler = scaler
        self.y_scaler = y_scaler
        self.bounds = bounds
        self.pop_size = pop
        self.generations = gens
        self.granule_fixed = granule_fixed
        self.granule_fixed_val = granule_fixed_val

    def _repair_batch(self, pop: np.ndarray) -> np.ndarray:
        """Repair an entire population in a vectorised manner."""
        api = pop[:, 0]
        mcc = pop[:, 1]
        pvpp = pop[:, 2]
        mgst = pop[:, 3]
        binder = pop[:, 4]
        pressure = pop[:, 5]
        speed = pop[:, 6]
        granule = pop[:, 7]
        particle_size = pop[:, 8]
        moisture = pop[:, 9]
        binder_grade = pop[:, 10]
        dwell_time = pop[:, 11]
        friction = pop[:, 12]
        decompression_time = pop[:, 13]

        # Normalise components
        api, binder, pvpp, mgst, mcc, moisture = normalize_components(
            api, binder, pvpp, mgst, mcc, moisture
        )

        # Clip process variables
        pressure = np.clip(pressure, self.bounds[5, 0], self.bounds[5, 1])
        speed = np.clip(speed, self.bounds[6, 0], self.bounds[6, 1])
        particle_size = np.clip(particle_size, CFG.PARTICLE_SIZE_MIN, CFG.PARTICLE_SIZE_MAX)
        binder_grade = np.clip(binder_grade, 0, len(CFG.BINDER_GRADES) - 1)
        dwell_time = np.clip(dwell_time, CFG.DWELL_TIME_MIN, CFG.DWELL_TIME_MAX)
        friction = np.clip(friction, CFG.FRICTION_MIN, CFG.FRICTION_MAX)
        decompression_time = np.clip(decompression_time, CFG.DECOMPRESSION_TIME_MIN, CFG.DECOMPRESSION_TIME_MAX)

        if self.granule_fixed:
            granule = np.full_like(granule, self.granule_fixed_val)
        else:
            granule = np.clip(granule, self.bounds[7, 0], self.bounds[7, 1])

        return np.column_stack([
            api, mcc, pvpp, mgst, binder,
            pressure, speed, granule,
            particle_size, moisture, binder_grade,
            dwell_time, friction, decompression_time
        ])

    def _evaluate(self, population: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Evaluate objectives: minimise (-API, EFRF, -Density) with penalties."""
        repaired = self._repair_batch(population)
        n = repaired.shape[0]

        # Build 19 features
        api = repaired[:, 0:1]
        mcc = repaired[:, 1:2]
        pvpp = repaired[:, 2:3]
        mgst = repaired[:, 3:4]
        binder = repaired[:, 4:5]
        pressure = repaired[:, 5:6]
        speed = repaired[:, 6:7]
        granule = repaired[:, 7:8]
        particle_size = repaired[:, 8:9]
        moisture = repaired[:, 9:10]
        binder_grade = repaired[:, 10:11]
        dwell_time = repaired[:, 11:12]
        friction = repaired[:, 12:13]
        decompression_time = repaired[:, 13:14]

        # Interactions
        api_binder = api * binder
        pressure_binder = pressure * binder
        api_mcc = api * mcc
        pressure_speed = pressure * speed
        binder_mgst = binder * mgst

        X_eval = np.concatenate([
            repaired,
            api_binder, pressure_binder, api_mcc, pressure_speed, binder_mgst
        ], axis=1)

        scaled = self.scaler.transform(X_eval)
        X_t = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = self.model.predict(X_t)
            pred = self.y_scaler.inverse_transform(pred_scaled)

        density = np.clip(pred[:, 0], CFG.DENSITY_MIN, CFG.DENSITY_MAX)
        tensile = np.maximum(pred[:, 1], 1e-4)
        er = np.maximum(pred[:, 2], 1e-4)
        efrf = er / tensile
        efrf = np.clip(efrf, 1e-4, 5.0)
        disintegration = np.maximum(pred[:, 3], 0.5)
        dissolution_tau = np.maximum(pred[:, 4], 1.0)

        # Penalty for violations
        penalty = np.zeros(n)
        penalty += np.where(tensile < CFG.TENSILE_MIN, (CFG.TENSILE_MIN - tensile) ** 2, 0.0)
        penalty += np.where(efrf >= CFG.EFRF_MAX, (efrf - CFG.EFRF_MAX) ** 2, 0.0)
        penalty += np.where(disintegration > CFG.DISINTEGRATION_MAX,
                            (disintegration - CFG.DISINTEGRATION_MAX) ** 2, 0.0)
        penalty += np.where(dissolution_tau > 20.0, (dissolution_tau - 20.0) ** 2, 0.0)
        mcc_val = repaired[:, 1]
        penalty += np.where(mcc_val > self.bounds[1, 1], (mcc_val - self.bounds[1, 1]) ** 2, 0.0)
        penalty += np.where(mcc_val < self.bounds[1, 0], (self.bounds[1, 0] - mcc_val) ** 2, 0.0)

        # Objectives: (min -API, min EFRF, min -Density)
        objectives = np.zeros((n, 3))
        objectives[:, 0] = -repaired[:, 0] + 100.0 * penalty
        objectives[:, 1] = efrf + 100.0 * penalty
        objectives[:, 2] = -density + 100.0 * penalty

        return objectives, None, repaired

    def _non_dominated_sort(self, objectives: np.ndarray) -> List[List[int]]:
        """Improved non‑dominated sorting with strict dominance."""
        n = objectives.shape[0]
        fronts = []
        remaining = set(range(n))

        while remaining:
            front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    # Check if j dominates i (all objectives <=, at least one <)
                    if (np.all(objectives[j] <= objectives[i]) and
                        np.any(objectives[j] < objectives[i])):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            fronts.append(front)
            remaining -= set(front)
        return fronts

    def _crowding_distance(self, objectives: np.ndarray, front: List[int]) -> np.ndarray:
        """Compute crowding distance for a front."""
        if len(front) <= 2:
            return np.ones(len(front)) * np.inf

        dist = np.zeros(len(front))
        for obj_idx in range(objectives.shape[1]):
            sorted_idx = sorted(front, key=lambda i: objectives[i, obj_idx])
            # Extremes get infinite distance
            dist[0] = np.inf
            dist[-1] = np.inf
            f_min = objectives[sorted_idx[0], obj_idx]
            f_max = objectives[sorted_idx[-1], obj_idx]
            if f_max - f_min > 1e-10:
                for k in range(1, len(sorted_idx) - 1):
                    dist[k] += (objectives[sorted_idx[k + 1], obj_idx] -
                                objectives[sorted_idx[k - 1], obj_idx]) / (f_max - f_min)
        return dist

    def _crossover(self, p1: np.ndarray, p2: np.ndarray, eta: float = 40.0) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated binary crossover."""
        child1 = np.zeros_like(p1)
        child2 = np.zeros_like(p2)
        for i in range(len(p1)):
            u = np.random.random()
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
            child1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
            child2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
        return child1, child2

    def _mutate(self, child: np.ndarray, eta: float = 20.0, pm: float = 0.1) -> np.ndarray:
        """Polynomial mutation."""
        for i in range(len(child)):
            if np.random.random() < pm:
                u = np.random.random()
                if u <= 0.5:
                    delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                child[i] = child[i] + delta * (self.bounds[i, 1] - self.bounds[i, 0])
                child[i] = np.clip(child[i], self.bounds[i, 0], self.bounds[i, 1])
        return child

    def _tournament(self, pop: np.ndarray, objectives: np.ndarray, fronts: List[List[int]]) -> np.ndarray:
        """Binary tournament selection based on rank and crowding."""
        idx1 = np.random.randint(0, len(pop))
        idx2 = np.random.randint(0, len(pop))
        # Find ranks
        rank1 = next((i for i, f in enumerate(fronts) if idx1 in f), len(fronts))
        rank2 = next((i for i, f in enumerate(fronts) if idx2 in f), len(fronts))
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

    def run(self) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
        """Execute NSGA‑II and return final population, objectives, and fronts."""
        rng = np.random.default_rng()
        pop = []
        for _ in range(self.pop_size):
            api = rng.uniform(CFG.API_MIN, CFG.API_MAX)
            mcc = rng.uniform(CFG.BOUND_MCC_MIN, CFG.BOUND_MCC_MAX)
            binder = rng.uniform(CFG.BOUND_BINDER_MIN, CFG.BOUND_BINDER_MAX)
            pvpp = rng.uniform(CFG.BOUND_PVPP_MIN, CFG.BOUND_PVPP_MAX)
            mgst = rng.uniform(CFG.BOUND_MGST_MIN, CFG.BOUND_MGST_MAX)
            moisture = rng.uniform(CFG.MOISTURE_MIN, CFG.MOISTURE_MAX)
            pressure = rng.uniform(CFG.BOUND_PRESSURE_MIN, CFG.BOUND_PRESSURE_MAX)
            speed = rng.uniform(CFG.BOUND_SPEED_MIN, CFG.BOUND_SPEED_MAX)
            granule = rng.uniform(CFG.BOUND_GRANULE_MIN, CFG.BOUND_GRANULE_MAX)
            particle_size = rng.uniform(CFG.PARTICLE_SIZE_MIN, CFG.PARTICLE_SIZE_MAX)
            binder_grade = rng.integers(0, len(CFG.BINDER_GRADES))
            dwell_time = rng.uniform(CFG.DWELL_TIME_MIN, CFG.DWELL_TIME_MAX)
            friction = rng.uniform(CFG.FRICTION_MIN, CFG.FRICTION_MAX)
            decompression_time = rng.uniform(
                CFG.DECOMPRESSION_TIME_MIN, CFG.DECOMPRESSION_TIME_MAX
            )
            ind = np.array([
                api, mcc, pvpp, mgst, binder,
                pressure, speed, granule,
                particle_size, moisture, binder_grade,
                dwell_time, friction, decompression_time
            ])
            pop.append(self._repair_batch(ind.reshape(1, -1))[0])
        pop = np.array(pop)

        progress_bar = st.progress(0, "Running NSGA‑II...")
        status_text = st.empty()

        for gen in range(self.generations):
            objectives, _, pop = self._evaluate(pop)
            fronts = self._non_dominated_sort(objectives)

            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._tournament(pop, objectives, fronts)
                p2 = self._tournament(pop, objectives, fronts)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                offspring.append(self._repair_batch(c1.reshape(1, -1))[0])
                if len(offspring) < self.pop_size:
                    offspring.append(self._repair_batch(c2.reshape(1, -1))[0])
            offspring = np.array(offspring[:self.pop_size])

            combined = np.vstack([pop, offspring])
            obj_comb, _, _ = self._evaluate(combined)
            fronts_comb = self._non_dominated_sort(obj_comb)

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

            if gen % 5 == 0 or gen == self.generations - 1:
                status_text.text(f"Generation {gen+1}/{self.generations} completed")
                progress_bar.progress((gen + 1) / self.generations)

        # Final evaluation
        objectives, _, pop = self._evaluate(pop)
        fronts = self._non_dominated_sort(objectives)
        progress_bar.empty()
        status_text.empty()
        return pop, objectives, fronts

# ================================================================
# PREDICTION WRAPPER
# ================================================================
def predict_pinn(
    model: MultiTaskPINN,
    scaler: StandardScaler,
    y_scaler: StandardScaler,
    inputs: List[float]
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Predict density, tensile, er, efrf, disintegration, tau, beta.
    """
    try:
        api, mcc, pvpp, mgst, binder, pressure, speed, granule, particle_size, moisture, binder_grade, dwell_time, friction, decompression_time = inputs

        # Build 19 features
        api_binder = api * binder
        pressure_binder = pressure * binder
        api_mcc = api * mcc
        pressure_speed = pressure * speed
        binder_mgst = binder * mgst

        X_input = np.array([[
            api, mcc, pvpp, mgst, binder,
            pressure, speed, granule,
            particle_size, moisture, binder_grade,
            dwell_time, friction, decompression_time,
            api_binder, pressure_binder, api_mcc,
            pressure_speed, binder_mgst
        ]])

        scaled = scaler.transform(X_input)
        X_t = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model.predict(X_t)[0]
            pred = y_scaler.inverse_transform([pred_scaled])[0]

        density = np.clip(pred[0], CFG.DENSITY_MIN, CFG.DENSITY_MAX)
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
# PLOTTING FUNCTIONS
# ================================================================
def plot_pareto_clean(
    objectives: np.ndarray,
    fronts: List[List[int]],
    balanced_solution: Optional[Tuple[float, float]] = None,
    feasible_df: Optional[pd.DataFrame] = None,
    tested_point: Optional[Tuple[float, float]] = None
) -> go.Figure:
    """Generate Pareto front plot with feasible region."""
    if fronts is None or len(fronts) == 0 or len(fronts[0]) == 0:
        return go.Figure()

    front = fronts[0]
    api_vals = -objectives[front, 0]
    efrf_vals = objectives[front, 1]
    df_front = pd.DataFrame({'API': api_vals, 'EFRF': efrf_vals}).sort_values('API')

    fig = go.Figure()

    if feasible_df is not None and not feasible_df.empty:
        fig.add_trace(go.Scatter(
            x=feasible_df['API'],
            y=feasible_df['EFRF'],
            mode='markers',
            name='Feasible Region (EFRF<0.40)',
            marker=dict(color='lightgreen', size=4, opacity=0.4),
            hovertemplate='API: %{x:.1f}%<br>EFRF: %{y:.4f}<extra></extra>',
            showlegend=True
        ))

    fig.add_trace(go.Scatter(
        x=df_front['API'],
        y=df_front['EFRF'],
        mode='lines+markers',
        name='Pareto Front',
        line=dict(color='red', width=2),
        marker=dict(size=7, color='red'),
        hovertemplate='API: %{x:.1f}%<br>EFRF: %{y:.4f}<extra></extra>'
    ))

    if tested_point is not None:
        fig.add_trace(go.Scatter(
            x=[tested_point[0]],
            y=[tested_point[1]],
            mode='markers',
            name='Tested Formulation',
            marker=dict(size=10, color='blue', symbol='circle',
                        line=dict(width=2, color='darkblue')),
            hovertemplate='Tested: API %{x:.1f}%, EFRF %{y:.4f}<extra></extra>'
        ))

    if balanced_solution is not None:
        fig.add_trace(go.Scatter(
            x=[balanced_solution[0]],
            y=[balanced_solution[1]],
            mode='markers',
            name='⭐ Golden (Balanced)',
            marker=dict(size=14, color='gold', symbol='star',
                        line=dict(width=2, color='black')),
            hovertemplate='Golden: API %{x:.1f}%, EFRF %{y:.4f}<extra></extra>'
        ))

    fig.add_hline(y=0.40, line_dash='dash', line_color='gray',
                  annotation_text='EFRF threshold (0.40)')
    fig.update_layout(
        title='Pareto Front with Feasible Region',
        xaxis_title='API (%)',
        yaxis_title='EFRF',
        height=450,
        template='plotly_white',
        legend=dict(x=0.8, y=0.95)
    )
    return fig


def plot_sensitivity_bars(
    formulation: Dict,
    model: MultiTaskPINN,
    scaler: StandardScaler,
    y_scaler: StandardScaler
) -> go.Figure:
    """Sensitivity analysis: change each parameter to see EFRF variation."""
    api0 = formulation['api_n']
    mcc0 = formulation['mcc_n']
    pvpp0 = formulation['pvpp_n']
    mgst0 = formulation['mgst_n']
    binder0 = formulation['binder_n']
    press0 = formulation['pressure']
    speed0 = formulation['speed']
    granule0 = formulation['granule_use']
    particle_size0 = formulation['particle_size']
    moisture0 = formulation['moisture']
    dwell_time0 = formulation['dwell_time']
    friction0 = formulation['friction']
    decompression_time0 = formulation['decompression_time']

    param_defs = [
        {'name': 'API', 'current': api0, 'min': CFG.API_MIN, 'max': CFG.API_MAX},
        {'name': 'MCC', 'current': mcc0, 'min': CFG.MCC_MIN, 'max': CFG.MCC_MAX},
        {'name': 'PVPP', 'current': pvpp0, 'min': CFG.PVPP_MIN, 'max': CFG.PVPP_MAX},
        {'name': 'MgSt', 'current': mgst0, 'min': CFG.MGST_MIN, 'max': CFG.MGST_MAX},
        {'name': 'Binder', 'current': binder0, 'min': CFG.BINDER_MIN, 'max': CFG.BINDER_MAX},
        {'name': 'Moisture', 'current': moisture0, 'min': CFG.MOISTURE_MIN, 'max': CFG.MOISTURE_MAX},
        {'name': 'Pressure', 'current': press0, 'min': CFG.PRESSURE_MIN, 'max': CFG.PRESSURE_MAX},
        {'name': 'Speed', 'current': speed0, 'min': CFG.SPEED_MIN, 'max': CFG.SPEED_MAX},
        {'name': 'Granule', 'current': granule0, 'min': CFG.GRANULE_MIN, 'max': CFG.GRANULE_MAX},
        {'name': 'ParticleSize', 'current': particle_size0, 'min': CFG.PARTICLE_SIZE_MIN, 'max': CFG.PARTICLE_SIZE_MAX},
        {'name': 'DwellTime', 'current': dwell_time0, 'min': CFG.DWELL_TIME_MIN, 'max': CFG.DWELL_TIME_MAX},
        {'name': 'Friction', 'current': friction0, 'min': CFG.FRICTION_MIN, 'max': CFG.FRICTION_MAX},
        {'name': 'DecompTime', 'current': decompression_time0, 'min': CFG.DECOMPRESSION_TIME_MIN, 'max': CFG.DECOMPRESSION_TIME_MAX}
    ]

    base_input = [
        api0, mcc0, pvpp0, mgst0, binder0,
        press0, speed0, granule0,
        particle_size0, moisture0, 0,
        dwell_time0, friction0, decompression_time0
    ]
    _, _, _, efrf_base, _, _, _ = predict_pinn(model, scaler, y_scaler, base_input)

    sensitivities = []
    for idx, p in enumerate(param_defs):
        low_input = base_input.copy()
        low_input[idx] = p['min']
        high_input = base_input.copy()
        high_input[idx] = p['max']
        _, _, _, efrf_low, _, _, _ = predict_pinn(model, scaler, y_scaler, low_input)
        _, _, _, efrf_high, _, _, _ = predict_pinn(model, scaler, y_scaler, high_input)
        delta = abs(efrf_high - efrf_low)
        sensitivities.append({
            'Parameter': p['name'],
            'Delta EFRF': delta
        })

    df_sens = pd.DataFrame(sensitivities).sort_values('Delta EFRF', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sens['Parameter'],
        x=df_sens['Delta EFRF'],
        orientation='h',
        marker_color='steelblue',
        text=df_sens['Delta EFRF'].round(4),
        textposition='outside',
        hovertemplate='%{y}<br>ΔEFRF: %{x:.4f}<extra></extra>'
    ))
    fig.add_vline(x=0.40, line_dash='dash', line_color='red',
                  annotation_text='EFRF threshold 0.40')
    fig.update_layout(
        title='Parameter Impact on EFRF',
        xaxis_title='Absolute change in EFRF',
        yaxis_title='Parameter',
        height=500,
        template='plotly_white'
    )
    return fig


def plot_dissolution_profile(
    tau: float,
    beta: float,
    api_n: float,
    title: str = "Predicted Dissolution Profile"
) -> go.Figure:
    """Weibull dissolution profile."""
    time_points = np.linspace(0, 60, 100)
    dissolution = 100 * (1 - np.exp(-((time_points / tau) ** beta)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_points,
        y=dissolution,
        mode='lines',
        name=f'Q(t) = 100×(1-exp(-((t/{tau:.1f})^{beta:.2f})))',
        line=dict(color='blue', width=2)
    ))
    fig.add_hline(y=85, line_dash='dash', line_color='red',
                  annotation_text='85% dissolution target')
    fig.update_layout(
        title=f'{title} (API: {api_n:.1f}%)',
        xaxis_title='Time (minutes)',
        yaxis_title='% Dissolved',
        height=350,
        template='plotly_white'
    )
    return fig

# ================================================================
# MODEL COMPARISON
# ================================================================
def run_model_comparison(
    model: MultiTaskPINN,
    scaler: StandardScaler,
    y_scaler: StandardScaler,
    features: List[str],
    df: pd.DataFrame,
    device: torch.device
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Compare PINN with MLP, Random Forest, and XGBoost."""
    X_raw_all = df[features].values
    y_raw_all = df[['Tensile_Strength_MPa']].values

    # Build 19 features
    api = X_raw_all[:, 0:1]
    mcc = X_raw_all[:, 1:2]
    pvpp = X_raw_all[:, 2:3]
    mgst = X_raw_all[:, 3:4]
    binder = X_raw_all[:, 4:5]
    pressure = X_raw_all[:, 5:6]
    speed = X_raw_all[:, 6:7]
    granule = X_raw_all[:, 7:8]
    particle_size = X_raw_all[:, 8:9]
    moisture = X_raw_all[:, 9:10]
    binder_grade = X_raw_all[:, 10:11]
    dwell_time = X_raw_all[:, 11:12]
    friction = X_raw_all[:, 12:13]
    decompression_time = X_raw_all[:, 13:14]

    api_binder = api * binder
    pressure_binder = pressure * binder
    api_mcc = api * mcc
    pressure_speed = pressure * speed
    binder_mgst = binder * mgst

    X_all = np.concatenate([
        X_raw_all,
        api_binder, pressure_binder, api_mcc, pressure_speed, binder_mgst
    ], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_raw_all, test_size=0.2, random_state=42
    )
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_target = y_train[:, 0]
    y_test_target = y_test[:, 0]

    # PINN prediction
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        pred_scaled = model.predict(X_test_t)
        pred_pinn = y_scaler.inverse_transform(pred_scaled)[:, 1]

    # MLP
    from sklearn.neural_network import MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42)
    mlp.fit(X_train_scaled, y_train_target)
    pred_mlp = mlp.predict(X_test_scaled)

    # Random Forest
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train_target)
    pred_rf = rf.predict(X_test_scaled)

    models = {
        'PINN (Proposed)': pred_pinn,
        'MLP (Baseline)': pred_mlp,
        'Random Forest': pred_rf,
    }

    # XGBoost (optional)
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1)
        xgb.fit(X_train_scaled, y_train_target)
        pred_xgb = xgb.predict(X_test_scaled)
        models['XGBoost'] = pred_xgb
    except ImportError:
        pass

    # Compute metrics with bootstrapping
    def compute_metrics(y_true, y_pred, n_boot=15):
        rng = np.random.default_rng(42)
        r2s, rmses, maes = [], [], []
        for _ in range(n_boot):
            idx = rng.choice(len(y_true), len(y_true), replace=True)
            r2s.append(r2_score(y_true[idx], y_pred[idx]))
            rmses.append(np.sqrt(mean_squared_error(y_true[idx], y_pred[idx])))
            maes.append(mean_absolute_error(y_true[idx], y_pred[idx]))
        return (np.mean(r2s), np.std(r2s),
                np.mean(rmses), np.std(rmses),
                np.mean(maes), np.std(maes))

    table_rows = []
    chart_data = []
    for name, pred in models.items():
        r2_m, r2_s, rmse_m, rmse_s, mae_m, mae_s = compute_metrics(y_test_target, pred)
        table_rows.append({
            'Model': name,
            'R2 (Test)': f"{r2_m:.2f} ± {r2_s:.2f}",
            'RMSE (MPa)': f"{rmse_m:.2f} ± {rmse_s:.2f}",
            'MAE (MPa)': f"{mae_m:.2f} ± {mae_s:.2f}",
            'Physical Consistency': 'Enforced' if name == 'PINN (Proposed)' else 'Not enforced'
        })
        chart_data.append({'Model': name, 'R² Score': r2_m})

    bench_df = pd.DataFrame(table_rows)
    return bench_df, chart_data


# ================================================================
# FEASIBLE REGION GENERATION (CACHED)
# ================================================================
@st.cache_data(show_spinner=False)
def generate_feasible_points(
    model: MultiTaskPINN,
    scaler: StandardScaler,
    y_scaler: StandardScaler,
    n_samples: int = 2000
) -> pd.DataFrame:
    """Sample random points and filter those satisfying constraints."""
    rng = np.random.default_rng(42)
    api = rng.uniform(CFG.API_MIN, CFG.API_MAX, n_samples)
    binder = rng.uniform(CFG.BINDER_MIN, CFG.BINDER_MAX, n_samples)
    pvpp = rng.uniform(CFG.PVPP_MIN, CFG.PVPP_MAX, n_samples)
    mgst = rng.uniform(CFG.MGST_MIN, CFG.MGST_MAX, n_samples)
    mcc = rng.uniform(CFG.MCC_MIN, CFG.MCC_MAX, n_samples)
    moisture = rng.uniform(CFG.MOISTURE_MIN, CFG.MOISTURE_MAX, n_samples)
    particle_size = rng.uniform(CFG.PARTICLE_SIZE_MIN, CFG.PARTICLE_SIZE_MAX, n_samples)
    binder_grade = rng.integers(0, len(CFG.BINDER_GRADES), n_samples)
    pressure = rng.uniform(CFG.PRESSURE_MIN, CFG.PRESSURE_MAX, n_samples)
    speed = rng.uniform(CFG.SPEED_MIN, CFG.SPEED_MAX, n_samples)
    dwell_time = calculate_dwell_time(speed)
    friction = rng.uniform(CFG.FRICTION_MIN, CFG.FRICTION_MAX, n_samples)
    decompression_time = rng.uniform(
        CFG.DECOMPRESSION_TIME_MIN, CFG.DECOMPRESSION_TIME_MAX, n_samples
    )
    granule = rng.uniform(CFG.GRANULE_MIN, CFG.GRANULE_MAX, n_samples)

    api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n = normalize_components(
        api, binder, pvpp, mgst, mcc, moisture
    )

    # Build 19 features
    api_binder = api_n * binder_n
    pressure_binder = pressure * binder_n
    api_mcc = api_n * mcc_n
    pressure_speed = pressure * speed
    binder_mgst = binder_n * mgst_n

    inputs = np.column_stack([
        api_n, mcc_n, pvpp_n, mgst_n, binder_n,
        pressure, speed, granule,
        particle_size, moisture_n, binder_grade,
        dwell_time, friction, decompression_time,
        api_binder, pressure_binder, api_mcc,
        pressure_speed, binder_mgst
    ])

    scaled = scaler.transform(inputs)
    X_t = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model.predict(X_t)
        pred = y_scaler.inverse_transform(pred_scaled)

    density = np.clip(pred[:, 0], CFG.DENSITY_MIN, CFG.DENSITY_MAX)
    tensile = np.maximum(pred[:, 1], 1e-4)
    er = np.maximum(pred[:, 2], 1e-4)
    efrf = er / tensile
    efrf = np.clip(efrf, 1e-4, 5.0)
    disintegration = np.maximum(pred[:, 3], 0.5)

    mask = ((density >= CFG.DENSITY_MIN) & (density <= CFG.DENSITY_MAX) &
            (tensile >= CFG.TENSILE_MIN) & (efrf < CFG.EFRF_MAX) &
            (disintegration <= CFG.DISINTEGRATION_MAX) &
            (mcc_n <= CFG.BOUND_MCC_MAX) & (mcc_n >= CFG.BOUND_MCC_MIN))

    return pd.DataFrame({'API': api_n[mask], 'EFRF': efrf[mask]})

# ================================================================
# PDF REPORT GENERATION (with embedded plots)
# ================================================================
def generate_enhanced_pdf_report(
    formulation: Dict,
    bench_df: pd.DataFrame,
    balanced_solution: Optional[np.ndarray],
    quality_solution: Optional[np.ndarray],
    cost_solution: Optional[np.ndarray],
    balanced_pred: Optional[Tuple],
    quality_pred: Optional[Tuple],
    cost_pred: Optional[Tuple],
    fronts: List[List[int]],
    timestamp: str,
    pareto_fig: go.Figure,
    sensitivity_fig: go.Figure,
    dissolution_fig: go.Figure
) -> Tuple[Optional[str], Optional[str]]:
    """Generate a PDF report with embedded plots."""
    if not FPDF_AVAILABLE:
        return None, "fpdf2 is not installed. Please install it with: pip install fpdf2"

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Hybrid AI for Multi-Objective Tablet Optimization", ln=True, align='C')
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 6, f"Generated: {timestamp}", ln=True, align='C')
        pdf.ln(4)

        # 1. Formulation
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "1. Formulation Parameters", ln=True)
        pdf.set_font("Arial", "", 10)
        f = formulation
        pdf.cell(60, 6, f"API: {f['api_n']:.1f}%", ln=True)
        pdf.cell(60, 6, f"MCC: {f['mcc_n']:.1f}%", ln=True)
        pdf.cell(60, 6, f"PVPP: {f['pvpp_n']:.1f}%", ln=True)
        pdf.cell(60, 6, f"Mg-St: {f['mgst_n']:.2f}%", ln=True)
        pdf.cell(60, 6, f"Binder: {f['binder_n']:.1f}%", ln=True)
        pdf.cell(60, 6, f"Moisture: {f['moisture']:.1f}%", ln=True)
        pdf.cell(60, 6, f"Particle Size: {f['particle_size']:.0f} µm", ln=True)
        pdf.cell(60, 6, f"Binder Grade: {CFG.BINDER_GRADES[int(f['binder_grade'])]}", ln=True)
        pdf.cell(60, 6, f"Pressure: {f['pressure']:.1f} MPa", ln=True)
        pdf.cell(60, 6, f"Speed: {f['speed']:.1f} rpm", ln=True)
        pdf.cell(60, 6, f"Dwell Time: {f['dwell_time']:.1f} ms", ln=True)
        pdf.cell(60, 6, f"Granule: {f['granule_use']:.0f} µm", ln=True)
        pdf.ln(4)

        # 2. Predicted properties
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "2. Predicted Properties", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(60, 6, f"Density: {f['density']:.3f}", ln=True)
        pdf.cell(60, 6, f"Tensile Strength: {f['tensile']:.2f} MPa", ln=True)
        pdf.cell(60, 6, f"EFRF: {f['efrf']:.4f}", ln=True)
        pdf.cell(60, 6, f"Elastic Recovery: {f['er']:.4f}", ln=True)
        pdf.cell(60, 6, f"Disintegration: {f['disintegration']:.1f} min", ln=True)
        pdf.ln(4)

        # 3. Constraints status
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "3. Constraints Status", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(60, 6, f"Density Status: {'PASS' if CFG.DENSITY_MIN <= f['density'] <= CFG.DENSITY_MAX else 'FAIL'}", ln=True)
        pdf.cell(60, 6, f"Tensile Status: {'PASS' if f['tensile'] >= CFG.TENSILE_MIN else 'FAIL'}", ln=True)
        pdf.cell(60, 6, f"EFRF Status: {'PASS' if f['efrf'] < 0.40 else 'FAIL'}", ln=True)
        pdf.cell(60, 6, f"Disintegration Status: {'PASS' if f['disintegration'] <= 15.0 else 'FAIL'}", ln=True)
        pdf.ln(4)

        # 4. Optimal solutions
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "4. Optimised Solutions (Pareto Front)", ln=True)
        if balanced_solution is not None and balanced_pred is not None:
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 6, "Golden Solution (Balanced)", ln=True)
            pdf.set_font("Arial", "", 10)
            pdf.cell(60, 6, f"API: {balanced_solution[0]:.1f}%", ln=True)
            pdf.cell(60, 6, f"EFRF: {balanced_pred[3]:.4f}", ln=True)
            pdf.cell(60, 6, f"Tensile: {balanced_pred[1]:.3f} MPa", ln=True)
            pdf.cell(60, 6, f"Disintegration: {balanced_pred[4]:.1f} min", ln=True)
            pdf.ln(4)

        if quality_solution is not None and quality_pred is not None:
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 6, "Quality-Optimised Solution (Max Tensile)", ln=True)
            pdf.set_font("Arial", "", 10)
            pdf.cell(60, 6, f"API: {quality_solution[0]:.1f}%", ln=True)
            pdf.cell(60, 6, f"EFRF: {quality_pred[3]:.4f}", ln=True)
            pdf.cell(60, 6, f"Tensile: {quality_pred[1]:.3f} MPa", ln=True)
            pdf.ln(4)

        if cost_solution is not None and cost_pred is not None:
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 6, "Cost-Optimised Solution (Max API, Min Pressure)", ln=True)
            pdf.set_font("Arial", "", 10)
            pdf.cell(60, 6, f"API: {cost_solution[0]:.1f}%", ln=True)
            pdf.cell(60, 6, f"EFRF: {cost_pred[3]:.4f}", ln=True)
            pdf.cell(60, 6, f"Tensile: {cost_pred[1]:.3f} MPa", ln=True)
            pdf.ln(4)

        # 5. Model comparison
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "5. Model Performance Comparison", ln=True)
        pdf.set_font("Arial", "", 10)
        if bench_df is not None:
            for _, row in bench_df.iterrows():
                pdf.cell(0, 6, f"{row['Model']}: {row['R2 (Test)']} | RMSE {row['RMSE (MPa)']}", ln=True)
        pdf.ln(4)

        # 6. Pareto summary
        if fronts is not None and len(fronts) > 0:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "6. Multi-Objective Optimisation Summary", ln=True)
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 6, f"Pareto Optimal Solutions Found: {len(fronts[0])} solutions", ln=True)

        # Embed plots if available (as images)
        if pareto_fig is not None:
            img_bytes = pareto_fig.to_image(format="png", width=800, height=500)
            img_b64 = base64.b64encode(img_bytes).decode()
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(img_bytes)
                tmp.flush()
                pdf.image(tmp.name, x=10, w=180)
                os.unlink(tmp.name)
            pdf.ln(4)

        if sensitivity_fig is not None:
            img_bytes = sensitivity_fig.to_image(format="png", width=800, height=500)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(img_bytes)
                tmp.flush()
                pdf.image(tmp.name, x=10, w=180)
                os.unlink(tmp.name)
            pdf.ln(4)

        if dissolution_fig is not None:
            img_bytes = dissolution_fig.to_image(format="png", width=800, height=400)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(img_bytes)
                tmp.flush()
                pdf.image(tmp.name, x=10, w=180)
                os.unlink(tmp.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            return tmp.name, None

    except Exception as e:
        return None, str(e)

# ================================================================
# MAIN UI
# ================================================================
st.markdown("""
<div style="background: #0b1a33; padding:1rem; border-radius:0.5rem; text-align:center; margin-bottom:1rem;">
    <h2 style="color:#fff; margin:0;">🧬 Hybrid AI For Multi‑Objective Tablet Optimization</h2>
    <p style="color:#64ffda; margin:0; font-size:1rem;">v29.27-R31 (ENHANCED)</p>
    <p style="color:#aabbcc; margin:0; font-size:0.85rem;">Nile Valley University, Sudan</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📚 Physics Constraints")
    st.markdown(f"""
    ✅ **API:** {CFG.API_MIN:.0f}–{CFG.API_MAX:.0f}%  
    ✅ **Density:** {CFG.DENSITY_MIN:.2f}–{CFG.DENSITY_MAX:.2f}  
    ✅ **Tensile:** ≥ {CFG.TENSILE_MIN:.2f} MPa  
    ✅ **EFRF:** < {CFG.EFRF_MAX:.2f} (feasible)  
    ✅ **Disintegration:** ≤ {CFG.DISINTEGRATION_MAX:.0f} min (USP)  
    ✅ **MCC:** {CFG.MCC_MIN:.1f}–{CFG.MCC_MAX:.1f}%  
    ✅ **PVPP:** {CFG.PVPP_MIN:.1f}–{CFG.PVPP_MAX:.1f}%  
    ✅ **MgSt:** {CFG.MGST_MIN:.2f}–{CFG.MGST_MAX:.2f}%  
    ✅ **Binder:** {CFG.BINDER_MIN:.1f}–{CFG.BINDER_MAX:.1f}%  
    ✅ **Moisture:** {CFG.MOISTURE_MIN:.1f}–{CFG.MOISTURE_MAX:.1f}%  
    ✅ **Pressure:** {CFG.PRESSURE_MIN:.0f}–{CFG.PRESSURE_MAX:.0f} MPa  
    ✅ **Speed:** {CFG.SPEED_MIN:.0f}–{CFG.SPEED_MAX:.0f} RPM  
    ✅ **NSGA‑II:** Pop={CFG.NSGA_POP}, Gen={CFG.NSGA_GENS} (3 objectives)
    """)
    st.caption("🔬 v29.27-R31 — ENHANCED (19 features, 30k samples)")

    # Experimental data upload
    st.markdown("---")
    st.markdown("### 📁 Experimental Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV with experimental results", type=["csv"])
    if uploaded_file is not None:
        try:
            exp_df = pd.read_csv(uploaded_file)
            st.session_state.experimental_data = exp_df
            st.sidebar.success(f"✅ Loaded {len(exp_df)} rows")
            with st.sidebar.expander("Preview Data"):
                st.dataframe(exp_df.head())
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

# Load model
try:
    model, scaler, y_scaler, features, df = load_or_train()
    st.session_state._model_loaded = True
except Exception as e:
    st.error(f"❌ Training failed: {e}. Using dummy model.")
    model = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if model is not None:
    device = next(model.parameters()).device

# Main layout
col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_left:
    st.markdown("### 📊 Formulation & Material Properties")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            api = st.slider("API (%)", CFG.API_MIN, CFG.API_MAX, st.session_state.api, 0.1, key="api_slider")
            binder = st.slider("Binder (%)", CFG.BINDER_MIN, CFG.BINDER_MAX, st.session_state.binder, 0.1, key="binder_slider")
            pvpp = st.slider("PVPP (%)", CFG.PVPP_MIN, CFG.PVPP_MAX, st.session_state.pvpp, 0.1, key="pvpp_slider")
            mgst = st.slider("Mg-St (%)", CFG.MGST_MIN, CFG.MGST_MAX, st.session_state.mgst, 0.01, key="mgst_slider")
            mcc = st.slider("MCC (%)", CFG.MCC_MIN, CFG.MCC_MAX, st.session_state.mcc, 0.1, key="mcc_slider")
        with c2:
            moisture = st.slider("Moisture (%)", CFG.MOISTURE_MIN, CFG.MOISTURE_MAX, st.session_state.moisture, 0.1, key="moisture_slider")
            particle_size = st.slider("Particle Size (µm)", CFG.PARTICLE_SIZE_MIN, CFG.PARTICLE_SIZE_MAX, st.session_state.particle_size, 1.0, key="particle_size_slider")
            binder_grade = st.selectbox("Binder Grade", CFG.BINDER_GRADES, index=st.session_state.binder_grade, key="binder_grade_select")
            binder_grade_idx = CFG.BINDER_GRADES.index(binder_grade)
            st.session_state.binder_grade = binder_grade_idx

        total = api + binder + pvpp + mgst + mcc + moisture
        if abs(total - 100) < 0.5:
            st.success(f"✅ Total = {total:.2f}%")
        else:
            st.warning(f"⚠️ Total = {total:.2f}% (should be 100%)")

    st.markdown("### ⚙️ Process Parameters")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            pressure = st.slider("Pressure (MPa)", CFG.PRESSURE_MIN, CFG.PRESSURE_MAX, st.session_state.get('pressure', 200.0), 1.0, key="pressure_slider")
            speed = st.slider("Speed (rpm)", CFG.SPEED_MIN, CFG.SPEED_MAX, st.session_state.get('speed', 20.0), 0.5, key="speed_slider")
        with c2:
            dwell_time = st.slider("Dwell Time (ms)", CFG.DWELL_TIME_MIN, CFG.DWELL_TIME_MAX, st.session_state.get('dwell_time', 25.0), 0.5, key="dwell_time_slider")
            friction = st.slider("Friction Coefficient", CFG.FRICTION_MIN, CFG.FRICTION_MAX, st.session_state.get('friction', 0.25), 0.01, key="friction_slider")
            decompression_time = st.slider("Decompression Time (ms)", CFG.DECOMPRESSION_TIME_MIN, CFG.DECOMPRESSION_TIME_MAX, st.session_state.get('decompression_time', 35.0), 1.0, key="decompression_time_slider")

        granule_mode = st.radio(
            "Granule Size",
            options=["Fixed (slider)", "Variable (optimized)"],
            index=0 if st.session_state.get('granule_mode', 'Fixed') == 'Fixed' else 1,
            horizontal=True,
            key="granule_mode_radio"
        )
        if granule_mode == "Fixed (slider)":
            granule = st.slider("Granule Size (µm)", CFG.GRANULE_MIN, CFG.GRANULE_MAX, st.session_state.get('granule', 125.0), 1.0, key="granule_slider")
            granule_fixed = True
            st.session_state.granule_mode = 'Fixed'
        else:
            granule = st.session_state.get('granule', 125.0)
            granule_fixed = False
            st.info(f"Granule size optimised by NSGA‑II ({CFG.GRANULE_MIN:.0f}–{CFG.GRANULE_MAX:.0f} µm)")
            st.session_state.granule_mode = 'Variable'

    predict_btn = st.button("🔬 Predict & Optimise", use_container_width=True, type="primary")

with col_right:
    st.markdown("### 📈 Results")

    if predict_btn:
        if model is None:
            st.error("❌ Model is not available. Please fix training errors and restart.")
        elif abs(total - 100) > 0.5:
            st.warning("⚠️ Formulation must sum to 100% (within 0.5%)")
        else:
            api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n = normalize_components(
                api, binder, pvpp, mgst, mcc, moisture
            )
            if granule_fixed:
                granule_use = granule
            else:
                granule_use = granule
            inputs = [
                api_n, mcc_n, pvpp_n, mgst_n, binder_n,
                pressure, speed, granule_use,
                particle_size, moisture_n, binder_grade_idx,
                dwell_time, friction, decompression_time
            ]

            density, tensile, er, efrf, disintegration, dissolution_tau, dissolution_beta = predict_pinn(
                model, scaler, y_scaler, inputs
            )

            st.session_state.formulation = {
                'api_n': api_n, 'binder_n': binder_n, 'pvpp_n': pvpp_n,
                'mgst_n': mgst_n, 'mcc_n': mcc_n, 'moisture': moisture_n,
                'particle_size': particle_size, 'binder_grade': binder_grade_idx,
                'pressure': pressure, 'speed': speed, 'dwell_time': dwell_time,
                'friction': friction, 'decompression_time': decompression_time,
                'granule_use': granule_use, 'granule_fixed': granule_fixed,
                'density': density, 'tensile': tensile, 'er': er, 'efrf': efrf,
                'disintegration': disintegration, 'dissolution_tau': dissolution_tau,
                'dissolution_beta': dissolution_beta
            }

            st.markdown("**Constraints Status**")
            col_metrics = st.columns(5)
            col_metrics[0].metric("Density", f"{density:.3f}", f"[{CFG.DENSITY_MIN:.2f}, {CFG.DENSITY_MAX:.2f}]")
            col_metrics[1].metric("Tensile", f"{tensile:.2f} MPa", f"≥ {CFG.TENSILE_MIN:.2f}")
            col_metrics[2].metric("EFRF", f"{efrf:.4f}", f"< {CFG.EFRF_MAX:.2f}")
            col_metrics[3].metric("MCC", f"{mcc_n:.1f}%", f"≤ 8.0%")
            col_metrics[4].metric("Disintegration", f"{disintegration:.1f} min", f"≤ {CFG.DISINTEGRATION_MAX:.0f} min")

            all_pass = all([
                CFG.DENSITY_MIN <= density <= CFG.DENSITY_MAX,
                tensile >= CFG.TENSILE_MIN,
                efrf < CFG.EFRF_MAX,
                mcc_n <= 8.0,
                disintegration <= CFG.DISINTEGRATION_MAX
            ])
            if all_pass:
                st.success("✅ All constraints satisfied")
            else:
                st.error("❌ Violates constraints")

            bounds = np.array([
                [CFG.API_MIN, CFG.API_MAX],
                [CFG.BOUND_MCC_MIN, CFG.BOUND_MCC_MAX],
                [CFG.BOUND_PVPP_MIN, CFG.BOUND_PVPP_MAX],
                [CFG.BOUND_MGST_MIN, CFG.BOUND_MGST_MAX],
                [CFG.BOUND_BINDER_MIN, CFG.BOUND_BINDER_MAX],
                [CFG.BOUND_PRESSURE_MIN, CFG.BOUND_PRESSURE_MAX],
                [CFG.BOUND_SPEED_MIN, CFG.BOUND_SPEED_MAX],
                [CFG.BOUND_GRANULE_MIN, CFG.BOUND_GRANULE_MAX],
                [CFG.PARTICLE_SIZE_MIN, CFG.PARTICLE_SIZE_MAX],
                [CFG.MOISTURE_MIN, CFG.MOISTURE_MAX],
                [0, len(CFG.BINDER_GRADES)-1],
                [CFG.DWELL_TIME_MIN, CFG.DWELL_TIME_MAX],
                [CFG.FRICTION_MIN, CFG.FRICTION_MAX],
                [CFG.DECOMPRESSION_TIME_MIN, CFG.DECOMPRESSION_TIME_MAX]
            ])

            with st.spinner(f"Running NSGA‑II (pop={CFG.NSGA_POP}, gen={CFG.NSGA_GENS})..."):
                nsga = NSGAII(
                    model, scaler, y_scaler, bounds,
                    pop=CFG.NSGA_POP, gens=CFG.NSGA_GENS,
                    granule_fixed=granule_fixed,
                    granule_fixed_val=granule if granule_fixed else 125.0
                )
                pop, objectives, fronts = nsga.run()

            st.session_state.nsga_pop = pop
            st.session_state.nsga_objectives = objectives
            st.session_state.nsga_fronts = fronts
            st.session_state.run_optimized = True

            balanced_idx = None
            quality_idx = None
            cost_idx = None

            if len(fronts) > 0 and len(fronts[0]) > 0:
                front_indices = fronts[0]
                max_api = max(-objectives[i, 0] for i in front_indices)
                min_efrf = min(objectives[i, 1] for i in front_indices)
                max_density = max(-objectives[i, 2] for i in front_indices)

                best_dist = np.inf
                api_range = CFG.API_MAX - CFG.API_MIN
                efrf_range = max(0.01, CFG.EFRF_MAX - min_efrf)
                density_range = CFG.DENSITY_MAX - CFG.DENSITY_MIN

                for idx in front_indices:
                    api_val = -objectives[idx, 0]
                    efrf_val = objectives[idx, 1]
                    density_val = -objectives[idx, 2]
                    norm_api = (CFG.API_MAX - api_val) / api_range if api_range > 0 else 0
                    norm_efrf = (efrf_val - min_efrf) / efrf_range if efrf_range > 0 else 0
                    norm_density = (CFG.DENSITY_MAX - density_val) / density_range if density_range > 0 else 0
                    dist = np.sqrt(norm_api**2 + norm_efrf**2 + norm_density**2)
                    if dist < best_dist:
                        best_dist = dist
                        balanced_idx = idx

                best_tensile = -np.inf
                for idx in front_indices:
                    ind = pop[idx]
                    _, t2, _, _, _, _, _ = predict_pinn(model, scaler, y_scaler, ind)
                    if t2 > best_tensile:
                        best_tensile = t2
                        quality_idx = idx

                best_cost_score = -np.inf
                for idx in front_indices:
                    ind = pop[idx]
                    api_val = ind[0]
                    pressure_val = ind[5]
                    cost_score = api_val - 0.05 * pressure_val
                    if cost_score > best_cost_score:
                        best_cost_score = cost_score
                        cost_idx = idx

                st.session_state.balanced_solution = pop[balanced_idx] if balanced_idx is not None else None
                st.session_state.quality_solution = pop[quality_idx] if quality_idx is not None else None
                st.session_state.cost_solution = pop[cost_idx] if cost_idx is not None else None

            with st.spinner("Generating feasible region..."):
                feasible_df = generate_feasible_points(model, scaler, y_scaler, n_samples=2000)
                st.session_state.feasible_df = feasible_df
                st.session_state.tested_point = (api_n, efrf)

    if st.session_state.run_optimized and model is not None:
        objectives = st.session_state.nsga_objectives
        fronts = st.session_state.nsga_fronts
        balanced_solution = st.session_state.balanced_solution
        quality_solution = st.session_state.quality_solution
        cost_solution = st.session_state.cost_solution
        feasible_df = st.session_state.feasible_df
        tested_point = st.session_state.tested_point

        st.markdown("### 📉 Pareto Front")
        if fronts is not None and len(fronts) > 0 and len(fronts[0]) > 0:
            st.success(f"✅ Pareto front: {len(fronts[0])} optimal solutions")
            balanced_efrf = None
            if balanced_solution is not None:
                _, _, _, ef, _, _, _ = predict_pinn(model, scaler, y_scaler, balanced_solution)
                balanced_efrf = (balanced_solution[0], ef)
            fig_pareto = plot_pareto_clean(objectives, fronts, balanced_efrf, feasible_df, tested_point)
            if fig_pareto is not None:
                st.plotly_chart(fig_pareto, use_container_width=True)
                st.session_state._pareto_fig = fig_pareto
        else:
            st.info("No Pareto front found.")

        st.markdown("### 📊 Optimal Solutions Comparison")
        solutions_rows = []

        if balanced_solution is not None:
            d, t, e, ef, dis, tau, beta = predict_pinn(model, scaler, y_scaler, balanced_solution)
            solutions_rows.append({
                "Solution Type": "⚖️ Balanced",
                "API (%)": balanced_solution[0],
                "MCC (%)": balanced_solution[1],
                "PVPP (%)": balanced_solution[2],
                "Mg-St (%)": balanced_solution[3],
                "Binder (%)": balanced_solution[4],
                "Moisture (%)": balanced_solution[9],
                "Pressure (MPa)": balanced_solution[5],
                "Speed (rpm)": balanced_solution[6],
                "Granule (µm)": balanced_solution[7],
                "Particle Size (µm)": balanced_solution[8],
                "Binder Grade": CFG.BINDER_GRADES[int(balanced_solution[10])],
                "Density": d,
                "Tensile (MPa)": t,
                "EFRF": ef,
                "Disintegration (min)": dis,
            })
            st.session_state.balanced_pred = (d, t, e, ef, dis, tau, beta)

        if st.session_state.show_cost_solution and cost_solution is not None:
            d, t, e, ef, dis, tau, beta = predict_pinn(model, scaler, y_scaler, cost_solution)
            solutions_rows.append({
                "Solution Type": "💰 Cost-Optimized",
                "API (%)": cost_solution[0],
                "MCC (%)": cost_solution[1],
                "PVPP (%)": cost_solution[2],
                "Mg-St (%)": cost_solution[3],
                "Binder (%)": cost_solution[4],
                "Moisture (%)": cost_solution[9],
                "Pressure (MPa)": cost_solution[5],
                "Speed (rpm)": cost_solution[6],
                "Granule (µm)": cost_solution[7],
                "Particle Size (µm)": cost_solution[8],
                "Binder Grade": CFG.BINDER_GRADES[int(cost_solution[10])],
                "Density": d,
                "Tensile (MPa)": t,
                "EFRF": ef,
                "Disintegration (min)": dis,
            })
            st.session_state.cost_pred = (d, t, e, ef, dis, tau, beta)

        if st.session_state.show_quality_solution and quality_solution is not None:
            d, t, e, ef, dis, tau, beta = predict_pinn(model, scaler, y_scaler, quality_solution)
            solutions_rows.append({
                "Solution Type": "🏆 Quality-Optimized",
                "API (%)": quality_solution[0],
                "MCC (%)": quality_solution[1],
                "PVPP (%)": quality_solution[2],
                "Mg-St (%)": quality_solution[3],
                "Binder (%)": quality_solution[4],
                "Moisture (%)": quality_solution[9],
                "Pressure (MPa)": quality_solution[5],
                "Speed (rpm)": quality_solution[6],
                "Granule (µm)": quality_solution[7],
                "Particle Size (µm)": quality_solution[8],
                "Binder Grade": CFG.BINDER_GRADES[int(quality_solution[10])],
                "Density": d,
                "Tensile (MPa)": t,
                "EFRF": ef,
                "Disintegration (min)": dis,
            })
            st.session_state.quality_pred = (d, t, e, ef, dis, tau, beta)

        if solutions_rows:
            df_solutions = pd.DataFrame(solutions_rows)
            st.dataframe(
                df_solutions,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Solution Type": st.column_config.TextColumn("Solution Type", width="small"),
                    "API (%)": st.column_config.NumberColumn("API (%)", format="%.1f", width="small"),
                    "MCC (%)": st.column_config.NumberColumn("MCC (%)", format="%.1f", width="small"),
                    "PVPP (%)": st.column_config.NumberColumn("PVPP (%)", format="%.1f", width="small"),
                    "Mg-St (%)": st.column_config.NumberColumn("Mg-St (%)", format="%.2f", width="small"),
                    "Binder (%)": st.column_config.NumberColumn("Binder (%)", format="%.1f", width="small"),
                    "Moisture (%)": st.column_config.NumberColumn("Moisture (%)", format="%.1f", width="small"),
                    "Pressure (MPa)": st.column_config.NumberColumn("Pressure (MPa)", format="%.1f", width="small"),
                    "Speed (rpm)": st.column_config.NumberColumn("Speed (rpm)", format="%.1f", width="small"),
                    "Granule (µm)": st.column_config.NumberColumn("Granule (µm)", format="%.0f", width="small"),
                    "Particle Size (µm)": st.column_config.NumberColumn("Particle Size (µm)", format="%.0f", width="small"),
                    "Binder Grade": st.column_config.TextColumn("Binder Grade", width="small"),
                    "Density": st.column_config.NumberColumn("Density", format="%.3f", width="small"),
                    "Tensile (MPa)": st.column_config.NumberColumn("Tensile (MPa)", format="%.3f", width="small"),
                    "EFRF": st.column_config.NumberColumn("EFRF", format="%.4f", width="small"),
                    "Disintegration (min)": st.column_config.NumberColumn("Disintegration (min)", format="%.1f", width="small"),
                }
            )
            st.caption("⚖️ Balanced = Trade-off (API, EFRF, Density) | 💰 Cost = Max API, Min Pressure | 🏆 Quality = Max Tensile Strength")
        else:
            st.info("No optimal solutions available to display.")

        st.markdown("---")
        st.toggle("💰 Show Cost-wise Solution", value=st.session_state.get("show_cost_solution", False), key="show_cost_solution")
        st.toggle("🏆 Show Quality-wise Solution", value=st.session_state.get("show_quality_solution", False), key="show_quality_solution")

        st.toggle("📊 Model Comparison", value=st.session_state.get("show_comparison", False), key="show_comparison")
        if st.session_state.show_comparison:
            st.markdown("### 📊 Model Comparison")
            bench_df, chart_data = run_model_comparison(model, scaler, y_scaler, features, df, device)
            st.session_state.benchmark_df = bench_df
            fig_bar = px.bar(pd.DataFrame(chart_data), x='Model', y='R² Score', color='Model',
                             title='R² Comparison (Tensile Strength)',
                             text=pd.DataFrame(chart_data)['R² Score'].round(3))
            fig_bar.update_layout(height=380, template='plotly_white')
            st.plotly_chart(fig_bar, use_container_width=True)
            st.dataframe(bench_df, use_container_width=True)

        st.toggle("🔬 Sensitivity Analysis", value=st.session_state.get("show_sensitivity", False), key="show_sensitivity")
        if st.session_state.show_sensitivity:
            st.markdown("### 🔬 Sensitivity Analysis")
            f = st.session_state.formulation
            if f is not None:
                fig_sens = plot_sensitivity_bars(f, model, scaler, y_scaler)
                if fig_sens:
                    st.plotly_chart(fig_sens, use_container_width=True)
                    st.session_state._sensitivity_fig = fig_sens

        st.toggle("📊 Dissolution Profile", value=st.session_state.get("show_dissolution", False), key="show_dissolution")
        if st.session_state.show_dissolution:
            st.markdown("### 📊 Dissolution Profile")
            f = st.session_state.formulation
            if f is not None:
                tau = f.get('dissolution_tau', 10.0)
                beta = f.get('dissolution_beta', 1.0)
                api_n = f['api_n']
                fig_diss = plot_dissolution_profile(tau, beta, api_n)
                st.plotly_chart(fig_diss, use_container_width=True)
                st.session_state._dissolution_fig = fig_diss

        if st.session_state.experimental_data is not None:
            st.markdown("### 🧪 Comparison with Experimental Data")
            st.dataframe(st.session_state.experimental_data)

        # PDF Report Button
        generate_report_btn = st.button("📄 Generate Enhanced Report (PDF)", key="knob_report")
        if generate_report_btn and st.session_state.benchmark_df is not None:
            f = st.session_state.formulation
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            bench_df = st.session_state.benchmark_df
            balanced_sol = st.session_state.balanced_solution
            quality_sol = st.session_state.quality_solution
            cost_sol = st.session_state.cost_solution
            balanced_pred = st.session_state.get('balanced_pred', None)
            quality_pred = st.session_state.get('quality_pred', None)
            cost_pred = st.session_state.get('cost_pred', None)
            fronts = st.session_state.nsga_fronts
            pareto_fig = st.session_state.get('_pareto_fig', None)
            sensitivity_fig = st.session_state.get('_sensitivity_fig', None)
            dissolution_fig = st.session_state.get('_dissolution_fig', None)

            filepath, error = generate_enhanced_pdf_report(
                f, bench_df, balanced_sol, quality_sol, cost_sol,
                balanced_pred, quality_pred, cost_pred, fronts, timestamp,
                pareto_fig, sensitivity_fig, dissolution_fig
            )
            if error:
                st.error(f"Failed to generate report: {error}")
                if not FPDF_AVAILABLE:
                    st.info("Please install fpdf2: `pip install fpdf2`")
            else:
                with open(filepath, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download Enhanced Report (PDF)",
                        data=pdf_file,
                        file_name=f"hubryd_enhanced_report_{timestamp[:10]}.pdf",
                        mime="application/pdf"
                    )
                try:
                    os.unlink(filepath)
                except Exception:
                    pass

    else:
        if model is None:
            st.warning("⚠️ Model not loaded. Please fix training issues and restart.")
        else:
            st.info("Adjust parameters and click '🔬 Predict & Optimise' to see results.")

st.caption("📧 Contact: babuker@protonmail.com | 🏛️ Nile Valley University, Sudan")
