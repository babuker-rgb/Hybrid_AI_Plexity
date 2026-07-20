"""
Hubryd AI – v29.27-R27 (MINIMAL – Guaranteed to Run)
Simplified Hybrid AI for Multi-Objective Optimization
- Uses ONLY API as variable
- All other parameters fixed
- Guaranteed to run without array ambiguity
Nile Valley University · Sudan
"""

import streamlit as st

# ================================================================
# MUST BE FIRST STREAMLIT COMMAND
# ================================================================
st.set_page_config(page_title="Hybrid AI – Minimal", layout="wide")

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

# ================================================================
# CONSTANTS – EXTREMELY SIMPLE
# ================================================================
D_MIN = 0.70
D_MAX = 0.99
TENSILE_MIN = 1.50

API_MIN = 80.0
API_MAX = 98.0

# Fixed parameters (no sliders)
FIXED_BINDER = 3.5
FIXED_PVPP = 2.0
FIXED_MGST = 0.5
FIXED_MCC = 3.5
FIXED_PRESSURE = 200.0
FIXED_SPEED = 20.0
FIXED_GRANULE = 125.0

# Training parameters
N_SAMPLES = 5000  # Smaller for faster testing
ADAM_EPOCHS = 200
PATIENCE = 30
NSGA_POP = 30
NSGA_GENS = 20
HIDDEN_SIZE = 64

# ================================================================
# SESSION STATE
# ================================================================
if 'api' not in st.session_state:
    st.session_state.update({
        'api': 90.5,
        'nsga_pop': None,
        'nsga_objectives': None,
        'nsga_fronts': None,
        'balanced_solution': None,
        'run_optimized': False,
        'formulation': None,
        'feasible_df': None,
        'tested_point': None,
    })

# ================================================================
# DATA GENERATION – NO ARRAYS IN CONDITIONS
# ================================================================
def generate_data(n_samples=N_SAMPLES, random_state=42):
    """Generate data with ONLY API as variable."""
    np.random.seed(random_state)
    
    # Only API varies
    api = np.random.uniform(API_MIN, API_MAX, n_samples)
    
    # All other parameters are fixed
    binder = np.full(n_samples, FIXED_BINDER)
    pvpp = np.full(n_samples, FIXED_PVPP)
    mgst = np.full(n_samples, FIXED_MGST)
    mcc = np.full(n_samples, FIXED_MCC)
    pressure = np.full(n_samples, FIXED_PRESSURE)
    speed = np.full(n_samples, FIXED_SPEED)
    granule = np.full(n_samples, FIXED_GRANULE)
    
    # Simple physics (Heckel – direct calculation)
    k = 0.025 + 0.0001 * FIXED_PRESSURE
    A = 1.0 + 0.01 * (api - 85.0) - 0.05 * FIXED_BINDER
    x_val = k * FIXED_PRESSURE + A
    D = 1.0 - np.exp(-x_val)
    D = np.clip(D, D_MIN, D_MAX)
    D += np.random.normal(0, 0.002, n_samples)
    D = np.clip(D, D_MIN, D_MAX)
    
    # Tensile strength
    porosity = 1.0 - D
    sigma0 = 5.0 + 0.1 * (api - 85.0) + 0.2 * FIXED_BINDER - 0.5 * FIXED_MGST
    sigma0 = np.clip(sigma0, 2.0, 8.0)
    b = 2.5 - 0.005 * (FIXED_PRESSURE - 80.0)
    b = np.clip(b, 1.5, 3.5)
    
    tensile_base = sigma0 * np.exp(-b * porosity)
    api_effect = 1.0 - 0.005 * (api - 85.0)
    binder_effect = 1.0 + 0.03 * (FIXED_BINDER - 2.0)
    mgst_effect = 1.0 - 0.1 * (FIXED_MGST - 0.2)
    pvpp_effect = 1.0 - 0.02 * (FIXED_PVPP - 3.0)
    speed_effect = 1.0 - 0.002 * (FIXED_SPEED - 10.0)
    
    strength = (tensile_base * api_effect * binder_effect *
                mgst_effect * pvpp_effect * speed_effect)
    strength *= np.random.normal(1.0, 0.01, n_samples)
    strength = np.clip(strength, 0.5, 6.0)
    
    # Elastic Recovery
    er_base = (1.8 + 0.3 * (api - 85.0)/10.0 +
               0.08 * (FIXED_SPEED - 10.0)/30.0 -
               0.1 * (FIXED_PRESSURE - 100.0)/150.0)
    er_base = er_base * (1.0 - 0.15 * (D - 0.4))
    er = er_base + np.random.normal(0, 0.01, n_samples)
    er = np.clip(er, 0.5, 4.0)
    
    # Build DataFrame
    X = np.column_stack([api, pressure, speed, granule])
    df = pd.DataFrame(X, columns=['API_%', 'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm'])
    df['Density'] = D
    df['Tensile_Strength_MPa'] = strength
    df['Elastic_Recovery_%'] = er
    
    return df, ['API_%', 'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm']

# ================================================================
# PINN MODEL – MINIMAL
# ================================================================
class MinimalPINN(nn.Module):
    def __init__(self, input_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )
    
    def forward(self, X):
        return self.net(X)
    
    def predict(self, X_scaled):
        self.eval()
        with torch.no_grad():
            if not isinstance(X_scaled, torch.Tensor):
                X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
            device = next(self.parameters()).device
            X_scaled = X_scaled.to(device)
            return self.forward(X_scaled).cpu().numpy()
    
    def compute_loss(self, X_scaled, X_raw, y_true, y_scaler):
        y_pred = self.forward(X_scaled)
        loss = nn.MSELoss()(y_pred, y_true)
        return loss

# ================================================================
# SIMPLE NSGA-II
# ================================================================
class SimpleNSGAII:
    def __init__(self, model, scaler, y_scaler, bounds, pop=NSGA_POP, gens=NSGA_GENS):
        self.model = model
        self.scaler = scaler
        self.y_scaler = y_scaler
        self.bounds = bounds
        self.pop_size = pop
        self.generations = gens
    
    def _evaluate(self, pop):
        n = pop.shape[0]
        objectives = np.zeros((n, 2))
        
        for i in range(n):
            api = pop[i, 0]
            pressure = pop[i, 1]
            speed = pop[i, 2]
            granule = pop[i, 3]
            
            inputs = np.array([api, pressure, speed, granule]).reshape(1, -1)
            scaled = self.scaler.transform(inputs)
            X_t = torch.tensor(scaled, dtype=torch.float32)
            
            with torch.no_grad():
                pred_scaled = self.model.predict(X_t)[0]
                pred = self.y_scaler.inverse_transform([pred_scaled])[0]
            
            density = np.clip(pred[0], D_MIN, D_MAX)
            tensile = max(pred[1], 1e-4)
            er = max(pred[2], 1e-4)
            efrf = er / tensile
            
            # Objective 1: maximize API (negative for minimization)
            objectives[i, 0] = -api
            # Objective 2: minimize EFRF
            objectives[i, 1] = efrf
            
            # Penalty for constraints
            if density < D_MIN or density > D_MAX:
                objectives[i, 0] += 1000
                objectives[i, 1] += 1000
            if tensile < TENSILE_MIN:
                objectives[i, 0] += 1000
                objectives[i, 1] += 1000
            if efrf >= 0.40:
                objectives[i, 0] += 1000
                objectives[i, 1] += 1000
        
        return objectives
    
    def _non_dominated_sort(self, objectives):
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
                    if (objectives[j,0] <= objectives[i,0] and objectives[j,1] <= objectives[i,1]) and \
                       (objectives[j,0] < objectives[i,0] or objectives[j,1] < objectives[i,1]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            fronts.append(front)
            remaining = [idx for idx in remaining if idx not in front]
        return fronts
    
    def _crossover(self, p1, p2):
        child1 = np.zeros(4)
        child2 = np.zeros(4)
        for i in range(4):
            if np.random.random() < 0.5:
                child1[i] = p1[i]
                child2[i] = p2[i]
            else:
                child1[i] = p2[i]
                child2[i] = p1[i]
        return child1, child2
    
    def _mutate(self, child):
        for i in range(4):
            if np.random.random() < 0.2:
                child[i] += np.random.normal(0, 0.02 * (self.bounds[i,1] - self.bounds[i,0]))
                child[i] = np.clip(child[i], self.bounds[i,0], self.bounds[i,1])
        return child
    
    def run(self):
        rng = np.random.default_rng()
        pop = []
        for _ in range(self.pop_size):
            api = rng.uniform(self.bounds[0,0], self.bounds[0,1])
            pressure = rng.uniform(self.bounds[1,0], self.bounds[1,1])
            speed = rng.uniform(self.bounds[2,0], self.bounds[2,1])
            granule = rng.uniform(self.bounds[3,0], self.bounds[3,1])
            pop.append([api, pressure, speed, granule])
        pop = np.array(pop)
        
        for gen in range(self.generations):
            objectives = self._evaluate(pop)
            fronts = self._non_dominated_sort(objectives)
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = pop[np.random.randint(0, self.pop_size)]
                p2 = pop[np.random.randint(0, self.pop_size)]
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                offspring.append(c1)
                if len(offspring) < self.pop_size:
                    offspring.append(c2)
            offspring = np.array(offspring[:self.pop_size])
            combined = np.vstack([pop, offspring])
            obj_comb = self._evaluate(combined)
            fronts_comb = self._non_dominated_sort(obj_comb)
            new_pop = []
            remaining = self.pop_size
            for front in fronts_comb:
                if len(front) <= remaining:
                    new_pop.extend(combined[front])
                    remaining -= len(front)
                else:
                    new_pop.extend(combined[front[:remaining]])
                    remaining = 0
                    break
            pop = np.array(new_pop)
        
        objectives = self._evaluate(pop)
        fronts = self._non_dominated_sort(objectives)
        return pop, objectives, fronts

# ================================================================
# PREDICTION
# ================================================================
def predict_pinn(model, scaler, y_scaler, inputs):
    if model is None:
        return 0.7, 2.0, 0.5, 0.25
    try:
        scaled = scaler.transform([inputs])
        X_t = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model.predict(X_t)[0]
        pred = y_scaler.inverse_transform([pred_scaled])[0]
        density = np.clip(pred[0], D_MIN, D_MAX)
        tensile = max(pred[1], 1e-4)
        er = max(pred[2], 1e-4)
        efrf = er / tensile
        return density, tensile, er, efrf
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.7, 2.0, 0.5, 0.25

# ================================================================
# MODEL TRAINING – SIMPLE
# ================================================================
CACHE_DIR = tempfile.gettempdir()
CHECKPOINT_PATH = os.path.join(CACHE_DIR, 'hubryd_minimal_eng.pt')

@st.cache_resource
def load_or_train():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            model = MinimalPINN(input_dim=4, hidden=HIDDEN_SIZE)
            model.load_state_dict(ckpt['model_state'])
            scaler = ckpt['scaler']
            y_scaler = ckpt['y_scaler']
            df = ckpt['df']
            features = ckpt['features']
            return model, scaler, y_scaler, features, df
        except Exception as e:
            st.warning(f"Cache load failed: {e}. Retraining...")
            if os.path.exists(CHECKPOINT_PATH):
                os.remove(CHECKPOINT_PATH)
    
    st.caption("🔄 Training minimal model (5k samples, up to 200 epochs)...")
    df, features = generate_data(N_SAMPLES)
    X_raw = df[features].values
    y = df[['Density', 'Tensile_Strength_MPa', 'Elastic_Recovery_%']].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    device = torch.device("cpu")
    st.caption(f"🖥️ Using device: {device}")
    
    model = MinimalPINN(input_dim=4, hidden=HIDDEN_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    best_val_loss = np.inf
    patience_counter = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(ADAM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = model.compute_loss(X_train_t, None, y_train_t, None)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = nn.MSELoss()(model(X_test_t), y_test_t).item()
        
        if epoch % 20 == 0:
            status_text.text(f"Epoch {epoch+1}/{ADAM_EPOCHS} - Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CACHE_DIR, 'best_model_minimal.pt'))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                st.info(f"Early stopping at epoch {epoch+1}")
                break
        
        progress_bar.progress((epoch+1)/ADAM_EPOCHS)
    
    if os.path.exists(os.path.join(CACHE_DIR, 'best_model_minimal.pt')):
        model.load_state_dict(torch.load(os.path.join(CACHE_DIR, 'best_model_minimal.pt'), map_location=device))
    
    st.success(f"✅ Best validation loss: {best_val_loss:.6f}")
    
    checkpoint = {
        'model_state': model.state_dict(),
        'scaler': scaler,
        'y_scaler': y_scaler,
        'features': features,
        'df': df
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    st.success("✅ Model trained and cached successfully!")
    
    return model, scaler, y_scaler, features, df

# ================================================================
# PLOTTING
# ================================================================
def plot_pareto_simple(objectives, fronts, balanced_solution=None, tested_point=None):
    if fronts is None or len(fronts) == 0 or len(fronts[0]) == 0:
        return None
    
    front = fronts[0]
    df_front = pd.DataFrame({
        'API': -objectives[front, 0],
        'EFRF': objectives[front, 1]
    }).sort_values('API')
    
    fig = go.Figure()
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
            marker=dict(size=12, color='gold', symbol='star', line=dict(width=2, color='black')),
            hovertemplate='Golden: API %{x:.1f}%, EFRF %{y:.4f}<extra></extra>'
        ))
    
    fig.add_hline(y=0.40, line_dash='dash', line_color='gray',
                  annotation_text='EFRF threshold (0.40)')
    fig.update_layout(
        title='Pareto Front (Minimal Model)',
        xaxis_title='API (%)',
        yaxis_title='EFRF',
        height=400,
        template='plotly_white'
    )
    return fig

# ================================================================
# MAIN UI
# ================================================================
st.markdown("""
<div style="background: #0b1a33; padding:1rem; border-radius:0.5rem; text-align:center; margin-bottom:1rem;">
    <h2 style="color:#fff; margin:0;">🧬 Hybrid AI · Minimal Model</h2>
    <p style="color:#64ffda; margin:0; font-size:0.9rem;">v29.27-R27 (MINIMAL – GUARANTEED)</p>
    <p style="color:#aabbcc; margin:0; font-size:0.85rem;">Nile Valley University, Sudan</p>
    <p style="color:#8899aa; font-size:0.75rem;">Only API varies · All other parameters fixed</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📚 Fixed Parameters")
    st.markdown(f"""
    ✅ **Binder:** {FIXED_BINDER:.1f}%  
    ✅ **PVPP:** {FIXED_PVPP:.1f}%  
    ✅ **Mg-St:** {FIXED_MGST:.2f}%  
    ✅ **MCC:** {FIXED_MCC:.1f}%  
    ✅ **Pressure:** {FIXED_PRESSURE:.0f} MPa  
    ✅ **Speed:** {FIXED_SPEED:.0f} RPM  
    ✅ **Granule:** {FIXED_GRANULE:.0f} µm  
    """)
    st.caption("🔬 v29.27-R27 — Minimal Model")

# Load model
try:
    model, scaler, y_scaler, features, df = load_or_train()
except Exception as e:
    st.error(f"Training failed: {e}. Using dummy model.")
    model = None

# Layout
col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_left:
    st.markdown("### 📊 Parameter")
    with st.container(border=True):
        api = st.slider("API (%)", API_MIN, API_MAX, st.session_state.api, 0.1, key="api_slider")
    
    predict_btn = st.button("🔬 Predict & Optimise", use_container_width=True, type="primary")

with col_right:
    st.markdown("### 📈 Results")
    
    if predict_btn:
        if model is None:
            st.error("❌ Model is not available. Please fix training errors and restart the app.")
        else:
            inputs = [api, FIXED_PRESSURE, FIXED_SPEED, FIXED_GRANULE]
            density, tensile, er, efrf = predict_pinn(model, scaler, y_scaler, inputs)
            
            st.session_state.formulation = {
                'api': api,
                'density': density,
                'tensile': tensile,
                'er': er,
                'efrf': efrf
            }
            
            st.markdown("**Constraints Status** (D: 0.70–0.99, Tensile ≥ 1.50, EFRF < 0.40)")
            col_metrics = st.columns(3)
            col_metrics[0].metric("Density", f"{density:.3f}", f"[{D_MIN:.2f}, {D_MAX:.2f}]")
            col_metrics[1].metric("Tensile", f"{tensile:.2f} MPa", f"≥ {TENSILE_MIN:.2f}")
            col_metrics[2].metric("EFRF", f"{efrf:.4f}", f"< 0.40")
            
            if all([D_MIN <= density <= D_MAX, tensile >= TENSILE_MIN, efrf < 0.40]):
                st.success("✅ All constraints satisfied")
            else:
                st.error("❌ Violates constraints")
            
            # --- NSGA-II ---
            bounds = np.array([
                [API_MIN, API_MAX],
                [150.0, 250.0],
                [15.0, 30.0],
                [30.0, 250.0]
            ])
            
            with st.spinner(f"Running NSGA‑II (pop={NSGA_POP}, gen={NSGA_GENS})..."):
                nsga = SimpleNSGAII(model, scaler, y_scaler, bounds)
                pop, objectives, fronts = nsga.run()
            
            st.session_state.nsga_pop = pop
            st.session_state.nsga_objectives = objectives
            st.session_state.nsga_fronts = fronts
            st.session_state.run_optimized = True
            
            # Extract balanced solution
            if len(fronts) > 0 and len(fronts[0]) > 0:
                front_indices = fronts[0]
                max_api = max(-objectives[i, 0] for i in front_indices)
                min_efrf = min(objectives[i, 1] for i in front_indices)
                best_dist = np.inf
                balanced_idx = None
                
                for idx in front_indices:
                    api_val = -objectives[idx, 0]
                    efrf_val = objectives[idx, 1]
                    norm_api = (max_api - api_val) / (max_api - API_MIN) if max_api > API_MIN else 0
                    norm_efrf = (efrf_val - min_efrf) / (0.40 - min_efrf) if 0.40 > min_efrf else 0
                    dist = norm_api**2 + norm_efrf**2
                    if dist < best_dist:
                        best_dist = dist
                        balanced_idx = idx
                
                if balanced_idx is not None:
                    st.session_state.balanced_solution = (pop[balanced_idx][0], objectives[balanced_idx][1])
            
            st.session_state.tested_point = (api, efrf)
    
    if st.session_state.run_optimized and model is not None:
        objectives = st.session_state.nsga_objectives
        fronts = st.session_state.nsga_fronts
        balanced_solution = st.session_state.balanced_solution
        tested_point = st.session_state.tested_point
        
        st.markdown("### 📉 Pareto Front")
        if fronts is not None and len(fronts) > 0 and len(fronts[0]) > 0:
            st.success(f"✅ Pareto front: {len(fronts[0])} optimal solutions")
            fig = plot_pareto_simple(objectives, fronts, balanced_solution, tested_point)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### ⭐ Golden Solution (Balanced)")
        if balanced_solution is not None:
            st.write(f"**API:** {balanced_solution[0]:.1f}%")
            st.write(f"**EFRF:** {balanced_solution[1]:.4f}")
        else:
            st.info("No balanced solution found.")
        
        st.markdown("### 📊 Tested Formulation")
        if tested_point is not None:
            st.write(f"**API:** {tested_point[0]:.1f}%")
            st.write(f"**EFRF:** {tested_point[1]:.4f}")
        
        # Show formulation details
        if st.session_state.formulation is not None:
            f = st.session_state.formulation
            st.markdown("### 📋 Formulation Details")
            st.write(f"API: {f['api']:.1f}%")
            st.write(f"Density: {f['density']:.3f}")
            st.write(f"Tensile Strength: {f['tensile']:.2f} MPa")
            st.write(f"Elastic Recovery: {f['er']:.4f}%")
            st.write(f"EFRF: {f['efrf']:.4f}")
    
    else:
        if model is None:
            st.warning("⚠️ Model not loaded. Please fix training issues and restart the app.")
        else:
            st.info("Adjust API and click '🔬 Predict & Optimise' to see results.")

st.caption("📧 Contact: babuker@protonmail.com | 🏛️ Nile Valley University, Sudan")
