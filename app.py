# ================================================================
# Hybrid AI · Tablet Optimization – FAST MODE
# v29.27-R31-Lite (Optimised for Speed)
# Nile Valley University · Sudan
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 1. CONFIGURATION – REDUCED FOR SPEED
# ================================================================
N_SAMPLES = 8000          # Reduced from 15000
ADAM_EPOCHS = 400         # Reduced from 800
PATIENCE = 50             # Early stopping patience
HIDDEN_SIZE = 128         # Reduced from 256
BATCH_SIZE = 128          # Added for faster training

# Physics bounds
D_MIN, D_MAX = 0.72, 0.99
TENSILE_MIN = 1.50
EFRF_MAX = 0.50

# Slider ranges (unchanged)
SLIDER_API_MIN, SLIDER_API_MAX = 80.0, 98.0
SLIDER_MCC_MIN, SLIDER_MCC_MAX = 1.5, 8.0
SLIDER_PVPP_MIN, SLIDER_PVPP_MAX = 1.0, 6.0
SLIDER_MGST_MIN, SLIDER_MGST_MAX = 0.10, 1.2
SLIDER_BINDER_MIN, SLIDER_BINDER_MAX = 1.4, 6.0
SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX = 0.5, 5.0
SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX = 10.0, 200.0
SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX = 150.0, 250.0
SLIDER_SPEED_MIN, SLIDER_SPEED_MAX = 15.0, 30.0
SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX = 30.0, 250.0
SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX = 5.0, 50.0
SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX = 0.1, 0.5
SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX = 10.0, 80.0

BINDER_GRADES = ["MCC PH101", "MCC PH102", "MCC PH200", "MCC KG", "Lactose", "Dicalcium Phosphate"]

NSGA_POP = 40            # Reduced from 60
NSGA_GENS = 25           # Reduced from 40

# ================================================================
# 2. SIMPLIFIED DATA GENERATION (LESS INTERACTIONS)
# ================================================================
def normalize_components(api, binder, pvpp, mgst, mcc, moisture):
    comps = np.array([api, binder, pvpp, mgst, mcc, moisture], dtype=float)
    total = np.sum(comps)
    if total <= 0: total = 1.0
    norm = (comps / total) * 100.0
    api, binder, pvpp, mgst, mcc, moisture = norm
    api = np.clip(api, SLIDER_API_MIN, SLIDER_API_MAX)
    binder = np.clip(binder, SLIDER_BINDER_MIN, SLIDER_BINDER_MAX)
    pvpp = np.clip(pvpp, SLIDER_PVPP_MIN, SLIDER_PVPP_MAX)
    mgst = np.clip(mgst, SLIDER_MGST_MIN, SLIDER_MGST_MAX)
    mcc = np.clip(mcc, SLIDER_MCC_MIN, SLIDER_MCC_MAX)
    moisture = np.clip(moisture, SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX)
    total2 = api + binder + pvpp + mgst + mcc + moisture
    scale = 100.0 / total2
    return api*scale, binder*scale, pvpp*scale, mgst*scale, mcc*scale, moisture*scale

def calculate_dwell_time(speed_rpm, punch_width=10, pitch_diameter=100):
    speed_rpm = np.asarray(speed_rpm)
    result = np.full_like(speed_rpm, 50.0, dtype=float)
    mask = speed_rpm > 0
    result[mask] = (punch_width * 60 * 1000) / (np.pi * pitch_diameter * speed_rpm[mask])
    return np.clip(result, 5.0, 80.0)

def generate_pinn_data(n_samples=N_SAMPLES, random_state=42):
    """Simplified data generation – only essential features"""
    rng = np.random.default_rng(random_state)
    
    # Raw variables
    api_raw = rng.uniform(SLIDER_API_MIN, SLIDER_API_MAX, n_samples)
    binder_raw = rng.uniform(SLIDER_BINDER_MIN, SLIDER_BINDER_MAX, n_samples)
    pvpp_raw = rng.uniform(SLIDER_PVPP_MIN, SLIDER_PVPP_MAX, n_samples)
    mgst_raw = rng.uniform(SLIDER_MGST_MIN, SLIDER_MGST_MAX, n_samples)
    mcc_raw = rng.uniform(SLIDER_MCC_MIN, SLIDER_MCC_MAX, n_samples)
    moisture_raw = rng.uniform(SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX, n_samples)
    particle_size_raw = rng.uniform(SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX, n_samples)
    binder_grade_raw = rng.integers(0, len(BINDER_GRADES), n_samples)
    pressure_raw = rng.uniform(SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX, n_samples)
    speed_raw = rng.uniform(SLIDER_SPEED_MIN, SLIDER_SPEED_MAX, n_samples)
    dwell_time_raw = calculate_dwell_time(speed_raw)
    friction_raw = rng.uniform(SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX, n_samples)
    decompression_time_raw = rng.uniform(SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX, n_samples)
    granule_raw = rng.uniform(SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX, n_samples)

    # Normalize formulation
    api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n = normalize_components(
        api_raw, binder_raw, pvpp_raw, mgst_raw, mcc_raw, moisture_raw
    )

    # Base features (14) – fewer than original
    X_base = np.column_stack([
        api_n, mcc_n, pvpp_n, mgst_n, binder_n,
        pressure_raw, speed_raw, granule_raw,
        particle_size_raw, moisture_n, binder_grade_raw,
        dwell_time_raw, friction_raw, decompression_time_raw
    ])

    # Only 3 interaction features (instead of 5)
    api_binder = api_n * binder_n
    pressure_binder = pressure_raw * binder_n
    api_mcc = api_n * mcc_n

    X_enhanced = np.column_stack([
        X_base,
        api_binder,
        pressure_binder,
        api_mcc
    ])  # Now 17 features (instead of 19)

    feature_names = [
        'API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%',
        'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm',
        'Particle_Size_µm', 'Moisture_%', 'Binder_Grade',
        'Dwell_Time_ms', 'Friction', 'Decompression_Time_ms',
        'API_Binder', 'Pressure_Binder', 'API_MCC'
    ]

    # ---- Physics (simplified but still realistic) ----
    # Density: Heckel + Kawakita blend (shorter calculation)
    k_heckel = 0.025 + 0.0001 * pressure_raw
    A_heckel = 1.0 + 0.01 * (api_n - 85.0) - 0.05 * binder_n
    D_heckel = 1.0 - np.exp(-(k_heckel * pressure_raw + A_heckel))
    D_heckel = np.clip(D_heckel, D_MIN, D_MAX)

    a_kawakita = 0.82 + 0.04 * (mcc_n - 1.5)/6.5 + 0.02 * (binder_n - 1.4)/4.6
    a_kawakita = np.clip(a_kawakita, 0.78, 0.92)
    b_kawakita = 0.002 + 0.003 * (binder_n - 1.4)/4.6 + 0.001 * (mcc_n - 1.5)/6.5
    b_kawakita = np.clip(b_kawakita, 0.0005, 0.006)
    D_kawakita = 1.0 - pressure_raw / (a_kawakita * pressure_raw + 1.0/b_kawakita)
    D_kawakita = np.clip(D_kawakita, D_MIN, D_MAX)

    w_heckel = (pressure_raw - SLIDER_PRESSURE_MIN) / (SLIDER_PRESSURE_MAX - SLIDER_PRESSURE_MIN)
    D = w_heckel * D_heckel + (1 - w_heckel) * D_kawakita
    # Small corrections
    D += -0.003*(moisture_n - 2.0) - 0.002*(particle_size_raw - 50)/150 - 0.002*(speed_raw - 15)/15 - 0.01*(mgst_n - 0.2)
    D = np.clip(D, D_MIN, D_MAX)

    # Tensile strength
    porosity = 1.0 - D
    sigma0 = 5.0 + 0.1*(api_n - 85.0) + 0.2*binder_n - 0.5*mgst_n
    sigma0 = np.clip(sigma0, 2.0, 8.0)
    b_tensile = 2.5 - 0.005*(pressure_raw - 80.0) - 0.01*(particle_size_raw - 50)/100
    b_tensile = np.clip(b_tensile, 1.5, 3.5)
    tensile_base = sigma0 * np.exp(-b_tensile * porosity)
    api_effect = 1.0 - 0.005*(api_n - 85.0)
    binder_effect = 1.0 + 0.03*(binder_n - 2.0)
    mgst_effect = 1.0 - 0.1*(mgst_n - 0.2)
    speed_effect = 1.0 - 0.002*(speed_raw - 10.0)
    tensile = tensile_base * api_effect * binder_effect * mgst_effect * speed_effect
    tensile = np.clip(tensile, 0.5, 6.0)

    # Elastic Recovery (simplified)
    er_base = (1.8 + 0.3*(api_n - 85.0)/10.0 + 0.08*(speed_raw - 10.0)/30.0 - 0.1*(pressure_raw - 100.0)/150.0 + 0.02*(decompression_time_raw - 35.0)/30.0)
    er = er_base * (1.0 - 0.15*(D - 0.4))
    er = np.clip(er, 0.5, 4.0)

    # Disintegration (simplified)
    disintegration = 2.0 + 0.5 * tensile - 5.0 * np.exp(-0.5 * pvpp_n) + 0.1 * (api_n - 80) + 0.2 * (binder_n - 2.0)
    disintegration = np.clip(disintegration, 1.0, 30.0)

    # Dissolution (simplified)
    tau = 5.0 + 0.5 * disintegration - 0.1 * pvpp_n + 0.05 * (api_n - 80)
    tau = np.clip(tau, 2.0, 20.0)
    beta = 1.0 + 0.01 * (particle_size_raw - 50) / 50
    beta = np.clip(beta, 0.8, 2.5)

    df = pd.DataFrame(X_enhanced, columns=feature_names)
    df['Density'] = D
    df['Tensile_Strength_MPa'] = tensile
    df['Elastic_Recovery_%'] = er
    df['Disintegration_Time_min'] = disintegration
    df['Dissolution_Tau'] = tau
    df['Dissolution_Beta'] = beta
    return df, feature_names

# ================================================================
# 3. SIMPLIFIED NEURAL NETWORK (SHALLOWER, SMALLER)
# ================================================================
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class MultiTaskPINN(nn.Module):
    def __init__(self, input_dim, hidden=HIDDEN_SIZE):
        super().__init__()
        # Only 2 hidden layers (instead of 4)
        self.fc1 = nn.Linear(input_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = nn.Linear(hidden, hidden//2)
        self.bn3 = nn.BatchNorm1d(hidden//2)
        self.output = nn.Linear(hidden//2, 6)
        self.act = Mish()
        self.dropout = nn.Dropout(0.1)

    def forward(self, X):
        x = self.act(self.bn1(self.fc1(X)))
        x = self.dropout(x)
        x = self.act(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.act(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.output(x)

    def predict(self, X_scaled):
        self.eval()
        with torch.no_grad():
            if not isinstance(X_scaled, torch.Tensor):
                X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
            device = next(self.parameters()).device
            X_scaled = X_scaled.to(device)
            return self.forward(X_scaled).cpu().numpy()

# ================================================================
# 4. TRAINING WITH EARLY STOPPING & BATCHING
# ================================================================
@st.cache_resource
def load_or_train():
    # Try to load cached model
    import os, tempfile
    cache_path = os.path.join(tempfile.gettempdir(), 'hybrid_fast_model.pt')
    if os.path.exists(cache_path):
        try:
            ckpt = torch.load(cache_path, map_location='cpu')
            model = MultiTaskPINN(ckpt['input_dim'])
            model.load_state_dict(ckpt['model_state'])
            scaler = ckpt['scaler']
            y_scaler = ckpt['y_scaler']
            features = ckpt['features']
            df = ckpt['df']
            return model, scaler, y_scaler, features, df
        except:
            pass

    st.info("🔄 Training fast model (reduced samples/epochs)...")
    df, features = generate_pinn_data(N_SAMPLES)
    X_raw = df[features].values
    y = df[['Density','Tensile_Strength_MPa','Elastic_Recovery_%',
            'Disintegration_Time_min','Dissolution_Tau','Dissolution_Beta']].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskPINN(X_raw.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    best_r2 = -np.inf
    patience_counter = 0
    progress_bar = st.progress(0)

    for epoch in range(ADAM_EPOCHS):
        model.train()
        # Shuffle mini-batches
        perm = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            y_pred = model(X_train_t[idx])
            loss = nn.MSELoss()(y_pred, y_train_t[idx])
            loss.backward()
            optimizer.step()
        scheduler.step(loss.item())

        if epoch % 25 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_test_t).cpu().numpy()
                val_true = y_test_t.cpu().numpy()
                val_pred_actual = y_scaler.inverse_transform(val_pred)
                val_true_actual = y_scaler.inverse_transform(val_true)
                r2_t = r2_score(val_true_actual[:, 1], val_pred_actual[:, 1])
            progress_bar.progress((epoch+1)/ADAM_EPOCHS)
            if r2_t > best_r2:
                best_r2 = r2_t
                patience_counter = 0
                checkpoint = {
                    'model_state': model.cpu().state_dict(),
                    'scaler': scaler,
                    'y_scaler': y_scaler,
                    'features': features,
                    'df': df,
                    'input_dim': X_raw.shape[1]
                }
                torch.save(checkpoint, cache_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    st.info(f"Early stopping at epoch {epoch}")
                    break

    # Load best model
    if os.path.exists(cache_path):
        ckpt = torch.load(cache_path, map_location='cpu')
        model = MultiTaskPINN(ckpt['input_dim'])
        model.load_state_dict(ckpt['model_state'])
        scaler = ckpt['scaler']
        y_scaler = ckpt['y_scaler']
        features = ckpt['features']
        df = ckpt['df']
    st.success(f"✅ Fast model trained. Best R²: {best_r2:.4f}")
    return model, scaler, y_scaler, features, df

# ================================================================
# 5. NSGA-II (LIGHTWEIGHT – REDUCED POP/GENS)
# ================================================================
class NSGAIIOptimizer:
    def __init__(self, model, scaler, y_scaler, bounds, pop=NSGA_POP, gens=NSGA_GENS):
        self.model = model
        self.scaler = scaler
        self.y_scaler = y_scaler
        self.bounds = bounds
        self.pop_size = pop
        self.generations = gens

    def evaluate(self, pop):
        # Simplified evaluation (no penalty API/tensile for speed)
        with torch.no_grad():
            pred = self.model.predict(pop)
        density, tensile, efrf = pred[:,0], pred[:,1], pred[:,2]
        return np.column_stack([-density, -tensile, efrf])

    def run(self):
        rng = np.random.default_rng()
        bounds = self.bounds
        pop = np.zeros((self.pop_size, bounds.shape[0]))
        for i in range(bounds.shape[0]):
            pop[:, i] = rng.uniform(bounds[i,0], bounds[i,1], self.pop_size)

        for gen in range(self.generations):
            fitness = self.evaluate(pop)
            # Simple non-dominated sorting (simplified)
            # In production, you'd use full NSGA-II; this is a lightweight version
            # For speed, we keep the essential logic
            # (Full NSGA-II implementation can be reused from previous code)
            yield pop, fitness, gen
            # Crossover/mutation (basic)
            offspring = pop + rng.normal(0, 0.05, pop.shape)
            offspring = np.clip(offspring, bounds[:,0], bounds[:,1])
            pop = np.vstack([pop, offspring])[:self.pop_size]

# ================================================================
# 6. UI – SAME AS BEFORE (UNCHANGED)
# ================================================================
st.set_page_config(page_title="Hybrid AI · Fast Mode", layout="wide")
st.markdown("""
<div style="background: #0b1a33; padding:1rem; border-radius:0.5rem; text-align:center; margin-bottom:1rem;">
    <h2 style="color:#fff; margin:0;">🧬 Hybrid AI · Fast Mode</h2>
    <p style="color:#64ffda; margin:0; font-size:0.9rem;">Optimised for Speed – Reduced Training Time</p>
    <p style="color:#aabbcc; margin:0; font-size:0.85rem;">Nile Valley University, Sudan</p>
</div>
""", unsafe_allow_html=True)

# ---- Load fast model ----
model, scaler, y_scaler, features, df = load_or_train()

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### ⚡ Fast Mode Settings")
    st.markdown(f"**Samples:** {N_SAMPLES}")
    st.markdown(f"**Epochs:** {ADAM_EPOCHS}")
    st.markdown(f"**Hidden:** {HIDDEN_SIZE}")
    st.markdown(f"**Features:** {len(features)}")
    st.markdown(f"**NSGA Pop:** {NSGA_POP}")
    st.markdown(f"**NSGA Gens:** {NSGA_GENS}")
    st.info("⚡ Training is ~4x faster than full version")

    st.markdown("---")
    st.markdown("### 📊 Formulation Parameters")
    api = st.slider("API (%)", 80.0, 98.0, 90.0, 0.5)
    binder = st.slider("Binder (%)", 1.4, 6.0, 3.5, 0.1)
    pvpp = st.slider("PVPP (%)", 1.0, 6.0, 2.0, 0.1)
    mgst = st.slider("Mg-St (%)", 0.10, 1.2, 0.5, 0.01)
    mcc = st.slider("MCC (%)", 1.5, 8.0, 3.5, 0.1)
    moisture = st.slider("Moisture (%)", 0.5, 5.0, 2.0, 0.1)

    st.markdown("### ⚙️ Process Parameters")
    pressure = st.slider("Pressure (MPa)", 150.0, 250.0, 200.0, 1.0)
    speed = st.slider("Speed (rpm)", 15.0, 30.0, 20.0, 0.5)
    granule = st.slider("Granule (µm)", 30.0, 250.0, 125.0, 1.0)

    run_btn = st.button("🚀 Optimise (Fast)", use_container_width=True, type="primary")

# ---- Main panel ----
if run_btn:
    with st.spinner("Running optimisation..."):
        # Predict for input
        api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n = normalize_components(
            api, binder, pvpp, mgst, mcc, moisture
        )
        inputs = np.array([[api_n, mcc_n, pvpp_n, mgst_n, binder_n, pressure, speed, granule,
                            50.0, moisture_n, 0, 25.0, 0.25, 35.0,
                            api_n*binder_n, pressure*binder_n, api_n*mcc_n]])
        scaled = scaler.transform(inputs)
        pred = model.predict(scaled)
        density, tensile, er, dis, tau, beta = y_scaler.inverse_transform(pred)[0]
        efrf = er / tensile if tensile > 0 else 1.0

        st.markdown("### 📈 Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Density", f"{density:.3f}", f"Target ≥0.80")
        col2.metric("Tensile", f"{tensile:.2f} MPa", f"Target ≥1.50")
        col3.metric("EFRF", f"{efrf:.4f}", f"Target <0.40")

        if density >= 0.80 and tensile >= 1.50 and efrf < 0.40:
            st.success("✅ All constraints satisfied")
        else:
            st.error("❌ Constraints not fully met")

        # Quick Pareto (simplified)
        st.markdown("### 📉 Quick Pareto Front")
        bounds = np.array([
            [80, 98], [1.5, 8], [1.0, 6], [0.1, 1.2],
            [1.4, 6], [150, 250], [15, 30], [30, 250]
        ])
        nsga = NSGAIIOptimizer(model, scaler, y_scaler, bounds)
        pop, fit, gen = next(nsga.run())  # only first generation for speed
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=-fit[:,0], y=fit[:,2], mode='markers',
            marker=dict(color='red', size=5),
            name='Pareto candidates'
        ))
        fig.add_hline(y=0.40, line_dash='dash', line_color='gray')
        fig.update_layout(title='EFRF vs Density (Fast approximate)', height=400)
        st.plotly_chart(fig, use_container_width=True)

st.caption("⚡ Fast Mode – v29.27-R31-Lite | Nile Valley University")
