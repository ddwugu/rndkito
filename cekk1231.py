"""
FOL - Finding Oil Losses
Pipeline Leak Detection System - Multi-Pipeline Edition
PT Pertamina EP Jambi Field

Developed by Research & Development
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy import interpolate  # ← TAMBAHAN INI BRO!
from datetime import datetime

class EnhancedLeakAnalyzer:
    """
    Enhanced Leak Detection v4.0
    Auto-adjust untuk jumlah sensor berapapun
    """
    
    def __init__(self, base_config, elevation_df=None):
        self.base_config = base_config
        self.elevation_df = elevation_df
        self.has_elevation_data = elevation_df is not None
    
    def predict(self, sensor_locations, normal_pressure, drop_pressure, sensor_names=None):
        """
        MAIN PREDICTION FUNCTION
        
        Input:
            sensor_locations: array lokasi sensor (KP in km) - bisa 3, 4, 9+ sensor
            normal_pressure: array tekanan normal (psi)
            drop_pressure: array tekanan drop (psi)
            sensor_names: list nama sensor (optional)
        
        Output:
            dict dengan hasil analisis lengkap
        """
        # Convert to arrays
        locations = np.array(sensor_locations)
        normal_p = np.array(normal_pressure)
        drop_p = np.array(drop_pressure)
        n_sensors = len(locations)
        
        # Validate
        if len(normal_p) != n_sensors:
            raise ValueError(f"normal_pressure harus {n_sensors} elemen (sesuai jumlah sensor)")
        if len(drop_p) != n_sensors:
            raise ValueError(f"drop_pressure harus {n_sensors} elemen (sesuai jumlah sensor)")
        
        # Generate sensor names if not provided
        if sensor_names is None:
            sensor_names = [f'Sensor {i+1} (KP {loc:.1f})' for i, loc in enumerate(locations)]
        
        # Get elevations for these sensor locations
        if self.has_elevation_data:
            elev_interp = interpolate.interp1d(
                self.elevation_df['distance_km'],
                self.elevation_df['elevation'],
                kind='cubic',
                fill_value='extrapolate'
            )
            elevations = elev_interp(locations)
        else:
            elevations = np.zeros(n_sensors)
        
        # Calculate metrics
        delta_p = normal_p - drop_p
        abs_delta_p = np.abs(delta_p)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pressure_ratio = abs_delta_p / np.abs(normal_p) * 100
        pressure_ratio = np.nan_to_num(pressure_ratio, 0.0)
        
        # Suspicion index
        suspicion_index = self._calculate_suspicion_index(abs_delta_p, pressure_ratio, n_sensors)
        
        # All detection methods
        susp_loc = self._suspicion_method(locations, suspicion_index)
        grad_loc = self._gradient_method(locations, normal_p, drop_p)
        interp_loc = self._interpolation_method(locations, abs_delta_p)
        weighted_loc = self._weighted_method(locations, suspicion_index)
        elev_loc = self._elevation_method(locations, normal_p, drop_p, elevations, n_sensors)
        
        # Weighted final estimate
        cfg = self.base_config
        final_estimate = (
            susp_loc * cfg['FINAL_ESTIMATE_WEIGHTS']['suspicion'] +
            interp_loc * cfg['FINAL_ESTIMATE_WEIGHTS']['interpolation'] +
            grad_loc * cfg['FINAL_ESTIMATE_WEIGHTS']['gradient'] +
            elev_loc * cfg['FINAL_ESTIMATE_WEIGHTS']['elevation'] +
            weighted_loc * cfg['FINAL_ESTIMATE_WEIGHTS']['weighted']
        )
        
        estimate_std = np.std([susp_loc, interp_loc, grad_loc, elev_loc, weighted_loc])
        
        # Confidence
        if estimate_std < 2:
            confidence = "VERY HIGH (95%+)"
        elif estimate_std < 4:
            confidence = "HIGH (90-95%)"
        elif estimate_std < 6:
            confidence = "HIGH (85-90%)"
        else:
            confidence = "MEDIUM (75-85%)"
        
        # Return results
        return {
            'final_estimate': final_estimate,
            'estimate_std': estimate_std,
            'confidence': confidence,
            'zones': {
                'focus': (final_estimate - 3, final_estimate + 3),
                'critical': (final_estimate - 5, final_estimate + 5),
                'primary': (final_estimate - 10, final_estimate + 10)
            },
            'top_sensor_idx': np.argmax(suspicion_index),
            'methods': {
                'suspicion': susp_loc,
                'interpolation': interp_loc,
                'gradient': grad_loc,
                'elevation': elev_loc,
                'weighted': weighted_loc
            },
            'sensor_data': {
                'locations': locations,
                'names': sensor_names,
                'elevations': elevations,
                'normal_pressure': normal_p,
                'drop_pressure': drop_p,
                'delta_pressure': delta_p,
                'abs_delta_pressure': abs_delta_p,
                'pressure_ratio': pressure_ratio,
                'suspicion_index': suspicion_index
            }
        }
    
    def _calculate_suspicion_index(self, abs_delta_p, pressure_ratio, n_sensors):
        cfg = self.base_config
        suspicion_index = np.zeros(n_sensors)
        
        for i in range(n_sensors):
            delta_factor = abs_delta_p[i]
            ratio_factor = pressure_ratio[i]
            
            if i > 0 and i < n_sensors - 1:
                neighbor_avg = (abs_delta_p[i-1] + abs_delta_p[i+1]) / 2
                neighbor_diff = abs_delta_p[i] - neighbor_avg
            elif i == 0 and n_sensors > 1:
                neighbor_diff = abs_delta_p[i] - abs_delta_p[i+1]
            elif i == n_sensors - 1 and n_sensors > 1:
                neighbor_diff = abs_delta_p[i] - abs_delta_p[i-1]
            else:
                neighbor_diff = 0
            
            neighbor_factor = max(0, neighbor_diff)
            
            suspicion_index[i] = (
                delta_factor * cfg['SUSPICION_WEIGHTS'][0] +
                ratio_factor * cfg['SUSPICION_WEIGHTS'][1] +
                neighbor_factor * cfg['SUSPICION_WEIGHTS'][2]
            )
        
        return suspicion_index
    
    def _suspicion_method(self, locations, suspicion_index):
        cfg = self.base_config
        top_idx = np.argmax(suspicion_index)
        location = locations[top_idx] + cfg['UPSTREAM_BIAS_PRIMARY']
        return max(location, locations[0])
    
    def _gradient_method(self, locations, normal_p, drop_p):
        cfg = self.base_config
        if len(locations) < 2:
            return locations[0]
        
        changes = []
        locs = []
        for i in range(len(locations) - 1):
            dist = locations[i+1] - locations[i]
            if dist > 0:
                norm_grad = (normal_p[i+1] - normal_p[i]) / dist
                drop_grad = (drop_p[i+1] - drop_p[i]) / dist
                changes.append(np.abs(norm_grad - drop_grad))
                locs.append((locations[i] + locations[i+1]) / 2)
        
        if not changes:
            return locations[0]
        
        max_idx = np.argmax(changes)
        return locs[max_idx] + cfg['UPSTREAM_BIAS_GRADIENT']
    
    def _interpolation_method(self, locations, abs_delta_p):
        cfg = self.base_config
        if len(locations) < 4:
            max_idx = np.argmax(abs_delta_p)
            return locations[max_idx] + cfg['UPSTREAM_BIAS_INTERP']
        
        try:
            f = interpolate.interp1d(locations, abs_delta_p, kind='cubic', fill_value='extrapolate')
            x_fine = np.linspace(locations.min(), locations.max(), 2000)
            y_fine = f(x_fine)
            peak_idx = np.argmax(y_fine)
            return x_fine[peak_idx] + cfg['UPSTREAM_BIAS_INTERP']
        except:
            max_idx = np.argmax(abs_delta_p)
            return locations[max_idx] + cfg['UPSTREAM_BIAS_INTERP']
    
    def _weighted_method(self, locations, suspicion_index):
        cfg = self.base_config
        total = np.sum(suspicion_index)
        if total == 0:
            return np.mean(locations)
        weighted_loc = np.sum(suspicion_index * locations) / total
        return weighted_loc + cfg['UPSTREAM_BIAS_WEIGHTED']
    
    def _elevation_method(self, locations, normal_p, drop_p, elevations, n_sensors):
        cfg = self.base_config
        
        if not self.has_elevation_data:
            return locations[np.argmax(np.abs(normal_p - drop_p))] + cfg['UPSTREAM_BIAS_PRIMARY']
        
        # Elevation-corrected pressure
        psi_per_meter = cfg['PSI_PER_METER'] * cfg['FLUID_DENSITY']
        ref_elev = elevations[0]
        elev_corr = (elevations - ref_elev) * psi_per_meter
        
        normal_corr = normal_p - elev_corr
        drop_corr = drop_p - elev_corr
        
        # Hydraulic anomaly
        anomaly_scores = np.zeros(n_sensors)
        for i in range(1, n_sensors):
            dist = locations[i] - locations[i-1]
            if dist > 0:
                exp_grad = (normal_corr[i-1] - normal_corr[i]) / dist
                act_grad = (drop_corr[i-1] - drop_corr[i]) / dist
                anom = abs(act_grad - exp_grad)
                anomaly_scores[i-1] += anom * 0.5
                anomaly_scores[i] += anom * 0.5
        
        max_idx = np.argmax(anomaly_scores)
        return locations[max_idx] + cfg['UPSTREAM_BIAS_PRIMARY']

# ============================================================================
# BASE CONFIGURATION
# ============================================================================
BASE_CONFIG = {
    'SUSPICION_WEIGHTS': [0.4, 0.35, 0.25],
    'UPSTREAM_BIAS_PRIMARY': -1.5,
    'UPSTREAM_BIAS_GRADIENT': -1.0,
    'UPSTREAM_BIAS_INTERP': -0.8,
    'UPSTREAM_BIAS_WEIGHTED': -1.2,
    'PSI_PER_METER': 0.433,
    'FLUID_DENSITY': 0.85,
    'FINAL_ESTIMATE_WEIGHTS': {
        'suspicion': 0.30,
        'interpolation': 0.25,
        'gradient': 0.20,
        'elevation': 0.15,
        'weighted': 0.10
    }
}

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="FOL - Pipeline Leak Detection",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* Main theme */
    .main {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(255, 107, 53, 0.3);
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: #fff;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    /* Pipeline card */
    .pipeline-card {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ff6b35;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    
    .pipeline-name {
        color: #ff6b35;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .pipeline-info {
        color: #e4e4e4;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Sensor input card */
    .sensor-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 107, 53, 0.2);
        margin-bottom: 1rem;
    }
    
    .sensor-title {
        color: #ff6b35;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5f8d 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #ff6b35;
        margin-top: 2rem;
        box-shadow: 0 8px 32px rgba(255, 107, 53, 0.4);
    }
    
    .result-title {
        color: #ff6b35;
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .result-location {
        background: rgba(255, 107, 53, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1.5rem 0;
        border: 2px solid #ff6b35;
    }
    
    .result-location-value {
        color: #ff6b35;
        font-size: 3rem;
        font-weight: 900;
        margin: 0;
    }
    
    .result-location-label {
        color: #e4e4e4;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1rem;
        margin: 0.5rem;
    }
    
    .confidence-very-high {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
    }
    
    .confidence-high {
        background: linear-gradient(135deg, #fdcb6e 0%, #f39c12 100%);
        color: white;
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
        color: white;
    }
    
    /* Zone card */
    .zone-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        border-left: 4px solid;
    }
    
    .zone-focus {
        border-left-color: #e74c3c;
    }
    
    .zone-critical {
        border-left-color: #e67e22;
    }
    
    .zone-primary {
        border-left-color: #f39c12;
    }
    
    .zone-title {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.3rem;
    }
    
    .zone-range {
        color: #e4e4e4;
        font-size: 0.95rem;
    }
    
    /* Info box */
    .info-box {
        background: rgba(52, 152, 219, 0.1);
        border-left: 4px solid #3498db;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(241, 196, 15, 0.1);
        border-left: 4px solid #f1c40f;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Method breakdown table */
    .method-table {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #95a5a6;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 16px rgba(255, 107, 53, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 53, 0.6);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #0f0f1e;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #ff6b35;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #f7931e;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PIPELINE CONFIGURATIONS
# ============================================================================
PIPELINE_CONFIGS = {
    "BJG-TPN": {
        "model_file": "bjg_model.sav",
        "total_length": 26.6,
        "sensors": [
            {"name": "I. KP 0 SPOT 1 FOL", "location": 0.0},
            {"name": "II. KP 7.14 SPOT 2 FOL", "location": 7.14},
            {"name": "III. KP 15.4 SPOT 3 FOL", "location": 15.4},
            {"name": "IV. KP 19.7 SPOT 4 FOL", "location": 19.7}
        ],
        "example_normal": [136.0, 112.14, 95.4, 37.1],
        "example_drop": [133.0, 110.1, 82.5, 34.5],
        "description": "Pipeline dari Betara Jambi (BJG) menuju Tempino (TPN)",
        "fluid_type": "Crude Oil"
    },
    "BTJ-TPN": {
        "model_file": "btj_model.sav",
        "total_length": 13.8,
        "sensors": [
            {"name": "I. KP 0 SP BTJ", "location": 0.0},
            {"name": "II. KP SIMP PS Gajah 5.5", "location": 5.5},
            {"name": "III. KP Sandtrap II 9.7", "location": 9.7}
        ],
        "example_normal": [190.2, 80.0, 34.5],
        "example_drop": [185.5, 75.0, 32.1],
        "description": "Pipeline dari Betara Jambi (BTJ) menuju Tempino (TPN)",
        "fluid_type": "Crude Oil"
    },
    "KAS-TPN": {
        "model_file": "kas_model.sav",
        "total_length": 23.2,
        "sensors": [
            {"name": "I. KP 0 SPOT 1 FOL", "location": 0.0},
            {"name": "II. KP 7.8 SPOT 2 FOL", "location": 7.8},
            {"name": "III. KP 15.4 SPOT 3 FOL", "location": 15.4},
            {"name": "IV. KP 23.2 SPOT 4 FOL", "location": 23.2}
        ],
        "example_normal": [150.727, 125.92, 84.0367, 43.2134],
        "example_drop": [143.778, 104.54, 64.3846, 40.0],
        "description": "Pipeline dari Kenali Asam (KAS) menuju Tempino (TPN)",
        "fluid_type": "Crude Oil"
    },
    "KTT-KAS": {
        "model_file": "ktt_model.sav",
        "total_length": 44.6,
        "sensors": [
            {"name": "I. KP 0 SPOT 1 FOL", "location": 0.0},
            {"name": "II. KP 44.6 SPOT 2 FOL", "location": 44.6}
        ],
        "example_normal": [190.2, 22.0],
        "example_drop": [165.4, 20.5],
        "description": "Pipeline dari Kenali Asam Timur (KTT) menuju Kenali Asam (KAS)",
        "fluid_type": "Crude Oil"
    },
    "SG-KAS": {
        "model_file": "sg_model.sav",
        "total_length": 11.2,
        "sensors": [
            {"name": "I. KP 0 SPOT 1 FOL", "location": 0.0},
            {"name": "II. KP 11.2 SPOT 2 FOL", "location": 11.2}
        ],
        "example_normal": [276.6, 20.0],
        "example_drop": [268.01, 18.0],
        "description": "Pipeline dari Sei Gelam (SG) menuju Kenali Asam (KAS)",
        "fluid_type": "Crude Oil"
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model_local(file_path):
    """Load model from local file"""
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model, None
    except FileNotFoundError:
        return None, f"Model file '{file_path}' tidak ditemukan. Pastikan file berada di folder yang sama dengan aplikasi."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def validate_inputs(normal_pressure, drop_pressure):
    """Validate pressure inputs"""
    errors = []
    warnings = []
    
    # Check if at least one pair is provided
    valid_pairs = sum(1 for n, d in zip(normal_pressure, drop_pressure) if n is not None and d is not None)
    if valid_pairs == 0:
        errors.append("❌ Minimal 1 sensor harus diisi (Normal & Drop pressure)")
    
    # Check for negative values and incomplete pairs
    for i, (n, d) in enumerate(zip(normal_pressure, drop_pressure)):
        if n is not None and n < 0:
            errors.append(f"❌ Sensor {i+1}: Normal pressure tidak boleh negatif")
        if d is not None and d < 0:
            errors.append(f"❌ Sensor {i+1}: Drop pressure tidak boleh negatif")
        
        # Check if both or none are provided
        if (n is None and d is not None) or (n is not None and d is None):
            warnings.append(f"⚠️ Sensor {i+1}: Isi kedua nilai (Normal & Drop) atau kosongkan keduanya (isi 0)")
    
    return errors, warnings

def filter_active_sensors(config, normal_pressure, drop_pressure):
    """Filter only active sensors (those with valid data)"""
    active_sensors = []
    active_locations = []
    active_names = []
    active_normal = []
    active_drop = []
    
    for i, (n, d) in enumerate(zip(normal_pressure, drop_pressure)):
        if n is not None and d is not None:
            active_sensors.append(i)
            active_locations.append(config["sensors"][i]["location"])
            active_names.append(config["sensors"][i]["name"])
            active_normal.append(n)
            active_drop.append(d)
    
    return {
        "indices": active_sensors,
        "locations": np.array(active_locations),
        "names": active_names,
        "normal": np.array(active_normal),
        "drop": np.array(active_drop)
    }

def format_confidence(confidence_str):
    """Format confidence with badge styling"""
    if "VERY HIGH" in confidence_str:
        badge_class = "confidence-very-high"
        icon = "🎯"
    elif "HIGH" in confidence_str:
        badge_class = "confidence-high"
        icon = "✅"
    else:
        badge_class = "confidence-medium"
        icon = "⚠️"
    
    return f'<span class="confidence-badge {badge_class}">{icon} {confidence_str}</span>'

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">🛢️ FOL - Finding Oil Losses</h1>
            <p class="header-subtitle">
                Pipeline Leak Detection System | PT Pertamina EP Jambi Field<br>
                <b>Research and Development Team</b>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/ff6b35/ffffff?text=FOL+System", use_container_width=True)
        st.markdown("---")
        
        st.markdown("### 📍 Select Pipeline")
        selected_pipeline = st.selectbox(
            "Pilih Jalur Pipa",
            options=list(PIPELINE_CONFIGS.keys()),
            format_func=lambda x: f"{x} ({PIPELINE_CONFIGS[x]['total_length']} km)"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ System Info")
        st.info(f"""
        **Pipeline:** {selected_pipeline}
        
        **Length:** {PIPELINE_CONFIGS[selected_pipeline]['total_length']} km
        
        **Sensors:** {len(PIPELINE_CONFIGS[selected_pipeline]['sensors'])}
        
        **Fluid:** {PIPELINE_CONFIGS[selected_pipeline]['fluid_type']}
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Features")
        st.markdown("""
        - ✅ Multi-condition prediction
        - ✅ Partial sensor data support
        - ✅ Real-time analysis
        - ✅ Elevation-corrected
        - ✅ Multiple detection methods
        """)
        
        st.markdown("---")
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%d %B %Y')}")
    
    # Main content
    config = PIPELINE_CONFIGS[selected_pipeline]
    
    # Pipeline info card
    st.markdown(f"""
        <div class="pipeline-card">
            <div class="pipeline-name">📍 {selected_pipeline}</div>
            <div class="pipeline-info">
                <b>Route:</b> {config['description']}<br>
                <b>Total Length:</b> {config['total_length']} km<br>
                <b>Available Sensors:</b> {len(config['sensors'])} units<br>
                <b>Fluid Type:</b> {config['fluid_type']}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
        <div class="info-box">
            💡 <b>Cara Penggunaan:</b><br>
            • Isi <b>Normal Pressure</b> dan <b>Drop Pressure</b> untuk sensor yang aktif<br>
            • Untuk sensor yang tidak aktif/mati, <b>biarkan nilai 0</b> atau kosongkan<br>
            • Minimal 1 sensor harus terisi untuk melakukan prediksi<br>
            • Sistem akan otomatis menyesuaikan analisis berdasarkan jumlah sensor aktif
        </div>
    """, unsafe_allow_html=True)
    
    # Sensor inputs
    st.markdown("---")
    st.markdown("## 📊 Sensor Data Input")
    
    # Create button row
    col_btn1, col_btn2, col_space = st.columns([1, 1, 2])
    
    with col_btn1:
        use_example = st.button("📝 Load Example Data", use_container_width=True)
    
    with col_btn2:
        clear_data = st.button("🗑️ Clear All Data", use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Initialize session state for values if not exists
    if 'sensor_values' not in st.session_state:
        st.session_state.sensor_values = {}
    
    # Create table header
    st.markdown("""
        <div style="background: rgba(255, 107, 53, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <div style="display: grid; grid-template-columns: 3fr 1fr 2fr 2fr; gap: 1rem; font-weight: 700; color: #ff6b35;">
                <div>Sensor Name</div>
                <div style="text-align: center;">KP (km)</div>
                <div style="text-align: center;">Normal Pressure (psi)</div>
                <div style="text-align: center;">Drop Pressure (psi)</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    normal_pressure = []
    drop_pressure = []
    
    # Sensor input rows
    for i, sensor in enumerate(config["sensors"]):
        # Create row container
        st.markdown(f"""
            <div class="sensor-card" style="padding: 0.8rem; margin-bottom: 0.5rem;">
                <div style="color: #ff6b35; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem;">
                    {sensor['name']}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Create columns for inputs
        col_name, col_km, col_normal, col_drop = st.columns([3, 1, 2, 2])
        
        with col_km:
            st.markdown(f"""
                <div style="text-align: center; padding: 0.5rem; background: rgba(255, 107, 53, 0.1); border-radius: 5px; margin-top: 0.3rem;">
                    <span style="font-weight: 700; color: #ff6b35; font-size: 1.1rem;">{sensor['location']}</span>
                </div>
            """, unsafe_allow_html=True)
        
        with col_normal:
            # Determine default value
            if use_example:
                default_normal = config["example_normal"][i]
            elif clear_data:
                default_normal = 0.0
            else:
                key = f"normal_{selected_pipeline}_{i}"
                default_normal = st.session_state.sensor_values.get(key, 0.0)
            
            normal_val = st.number_input(
                f"Normal {i+1}",
                min_value=0.0,
                max_value=500.0,
                value=default_normal,
                step=0.01,
                format="%.2f",
                key=f"normal_{selected_pipeline}_{i}",
                label_visibility="collapsed",
                help="Tekanan normal operasi (0 = sensor tidak aktif)"
            )
            
            # Save to session state
            st.session_state.sensor_values[f"normal_{selected_pipeline}_{i}"] = normal_val
            normal_pressure.append(normal_val if normal_val > 0 else None)
        
        with col_drop:
            # Determine default value
            if use_example:
                default_drop = config["example_drop"][i]
            elif clear_data:
                default_drop = 0.0
            else:
                key = f"drop_{selected_pipeline}_{i}"
                default_drop = st.session_state.sensor_values.get(key, 0.0)
            
            drop_val = st.number_input(
                f"Drop {i+1}",
                min_value=0.0,
                max_value=500.0,
                value=default_drop,
                step=0.01,
                format="%.2f",
                key=f"drop_{selected_pipeline}_{i}",
                label_visibility="collapsed",
                help="Tekanan saat terjadi penurunan (0 = sensor tidak aktif)"
            )
            
            # Save to session state
            st.session_state.sensor_values[f"drop_{selected_pipeline}_{i}"] = drop_val
            drop_pressure.append(drop_val if drop_val > 0 else None)
    
    # Validate inputs
    st.markdown("---")
    validation_errors, validation_warnings = validate_inputs(normal_pressure, drop_pressure)
    
    if validation_errors:
        for error in validation_errors:
            st.error(error)
    
    if validation_warnings:
        for warning in validation_warnings:
            st.warning(warning)
    
    # Show active sensors count
    active_count = sum(1 for n, d in zip(normal_pressure, drop_pressure) if n is not None and d is not None)
    if active_count > 0:
        st.success(f"✅ {active_count} sensor aktif dari {len(config['sensors'])} sensor tersedia")

    
    # Predict button
    col_predict, col_space = st.columns([1, 3])
    with col_predict:
        predict_button = st.button(
            "🔍 DETECT LEAK LOCATION",
            disabled=len(validation_errors) > 0,
            use_container_width=True
        )
    
    # Prediction
    if predict_button and len(validation_errors) == 0:
        with st.spinner("🔄 Loading model and analyzing data..."):
            # Load model from local file
            model_obj, error = load_model_local(config["model_file"])
            
            if model_obj is None:
                st.error(f"""
                    ❌ **Failed to load model**
                    
                    {error}
                    
                    **Checklist:**
                    - ✅ File `{config["model_file"]}` harus berada di folder yang sama dengan `fol_leak_detection.py`
                    - ✅ Pastikan file tidak corrupt
                    - ✅ Pastikan semua 5 model files sudah diupload: 
                      - bjg_model.sav
                      - btj_model.sav
                      - kas_model.sav
                      - ktt_model.sav
                      - sg_model.sav
                """)
                return
            
            # Initialize analyzer with BASE_CONFIG
            analyzer = EnhancedLeakAnalyzer(BASE_CONFIG)
            
            # Filter active sensors
            active_data = filter_active_sensors(config, normal_pressure, drop_pressure)
            
            if len(active_data["indices"]) == 0:
                st.error("❌ Tidak ada sensor yang terisi dengan lengkap!")
                return
            
            # Display active sensors info
            st.info(f"""
                ℹ️ **Analyzing with {len(active_data['indices'])} active sensor(s):**
                {', '.join([f"Sensor {i+1}" for i in active_data['indices']])}
            """)
            
            # Predict
            try:
                results = analyzer.predict(
                    sensor_locations=active_data["locations"],
                    normal_pressure=active_data["normal"],
                    drop_pressure=active_data["drop"],
                    sensor_names=active_data["names"]
                )
                
                # Display results
                st.markdown("---")
                st.markdown("""
                    <div class="result-card">
                        <div class="result-title">🎯 LEAK DETECTION RESULTS</div>
                """, unsafe_allow_html=True)
                
                # Main result
                st.markdown(f"""
                    <div class="result-location">
                        <div class="result-location-value">KP {results['final_estimate']:.2f}</div>
                        <div class="result-location-label">Estimated Leak Location (± {results['estimate_std']:.2f} km)</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Confidence
                st.markdown(f"""
                    <div style="text-align: center; margin: 1.5rem 0;">
                        {format_confidence(results['confidence'])}
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Inspection zones
                st.markdown("---")
                st.markdown("## 🎯 Priority Inspection Zones")
                
                col_z1, col_z2, col_z3 = st.columns(3)
                
                focus = results['zones']['focus']
                critical = results['zones']['critical']
                primary = results['zones']['primary']
                
                with col_z1:
                    st.markdown(f"""
                        <div class="zone-card zone-focus">
                            <div class="zone-title">🔴 FOCUS ZONE</div>
                            <div class="zone-range">KP {focus[0]:.1f} - {focus[1]:.1f}</div>
                            <div class="zone-range" style="margin-top: 0.5rem; font-weight: 700;">HIGHEST PRIORITY</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col_z2:
                    st.markdown(f"""
                        <div class="zone-card zone-critical">
                            <div class="zone-title">🟠 CRITICAL ZONE</div>
                            <div class="zone-range">KP {critical[0]:.1f} - {critical[1]:.1f}</div>
                            <div class="zone-range" style="margin-top: 0.5rem;">Secondary inspection</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col_z3:
                    st.markdown(f"""
                        <div class="zone-card zone-primary">
                            <div class="zone-title">🟡 PRIMARY ZONE</div>
                            <div class="zone-range">KP {primary[0]:.1f} - {primary[1]:.1f}</div>
                            <div class="zone-range" style="margin-top: 0.5rem;">Tertiary inspection</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Method breakdown
                st.markdown("---")
                st.markdown("## 📊 Detection Method Breakdown")
                
                methods_df = pd.DataFrame({
                    "Method": [
                        "Suspicion Index",
                        "Interpolation",
                        "Gradient Analysis",
                        "Elevation/Hydraulic",
                        "Weighted Average"
                    ],
                    "Location (KP)": [
                        f"{results['methods']['suspicion']:.2f}",
                        f"{results['methods']['interpolation']:.2f}",
                        f"{results['methods']['gradient']:.2f}",
                        f"{results['methods']['elevation']:.2f}",
                        f"{results['methods']['weighted']:.2f}"
                    ],
                    "Weight": ["30%", "25%", "20%", "15%", "10%"]
                })
                
                st.dataframe(methods_df, use_container_width=True, hide_index=True)
                
                # Sensor details
                st.markdown("---")
                st.markdown("## 📈 Active Sensor Details")
                
                sensor_data = results['sensor_data']
                sensor_df = pd.DataFrame({
                    "Sensor": sensor_data['names'],
                    "KP (km)": [f"{loc:.1f}" for loc in sensor_data['locations']],
                    "Elevation (m)": [f"{elev:.1f}" if analyzer.has_elevation_data else "N/A" 
                                     for elev in sensor_data['elevations']],
                    "Normal (psi)": [f"{p:.2f}" for p in sensor_data['normal_pressure']],
                    "Drop (psi)": [f"{p:.2f}" for p in sensor_data['drop_pressure']],
                    "ΔP (psi)": [f"{p:.2f}" for p in sensor_data['delta_pressure']],
                    "Suspicion Score": [f"{s:.2f}" for s in sensor_data['suspicion_index']]
                })
                
                st.dataframe(sensor_df, use_container_width=True, hide_index=True)
                
                # Most suspicious sensor
                top_idx = results['top_sensor_idx']
                st.markdown(f"""
                    <div class="warning-box">
                        ⚠️ <b>Most Suspicious Sensor:</b> {sensor_data['names'][top_idx]} 
                        (KP {sensor_data['locations'][top_idx]:.1f}) - 
                        Score: {sensor_data['suspicion_index'][top_idx]:.2f}
                    </div>
                """, unsafe_allow_html=True)
                
                # Export results
                st.markdown("---")
                st.markdown("## 💾 Export Results")
                
                # Create export data
                export_data = {
                    "Pipeline": selected_pipeline,
                    "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Active Sensors": len(active_data['indices']),
                    "Estimated Location (KP)": f"{results['final_estimate']:.2f}",
                    "Uncertainty (km)": f"{results['estimate_std']:.2f}",
                    "Confidence": results['confidence'],
                    "Focus Zone": f"KP {focus[0]:.1f} - {focus[1]:.1f}",
                    "Critical Zone": f"KP {critical[0]:.1f} - {critical[1]:.1f}",
                    "Primary Zone": f"KP {primary[0]:.1f} - {primary[1]:.1f}"
                }
                
                export_df = pd.DataFrame([export_data])
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name=f"FOL_LeakDetection_{selected_pipeline}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"""
                    ❌ **Prediction Error**
                    
                    {str(e)}
                    
                    Please check your input data and try again.
                """)
                st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div class="footer">
            <b>FOL - Finding Oil Losses</b> | Pipeline Leak Detection System<br>
            Developed by <b>Team UWAK PO</b> - PT Pertamina EP Jambi Field<br>
            🏆 SKK Migas IOC Digital Hackathon 2025 - 1st Place Winner<br>
            <br>
            Powered by Machine Learning & IoT Technology
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()