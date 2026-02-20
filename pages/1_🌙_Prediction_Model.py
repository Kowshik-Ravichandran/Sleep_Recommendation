import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
from datetime import datetime
from modules.theme import apply_theme

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Sleep Lab",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply Premium Theme (shared across all pages)
apply_theme()

# --- ADDITIONAL PAGE-SPECIFIC STYLING ---
st.markdown("""
<style>
    /* Form Section Headers */
    .section-header {
        font-size: 1rem;
        font-weight: 700;
        color: #6C63FF;
        margin: 0 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(108, 99, 255, 0.3);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Form Card */
    .form-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .form-card:hover {
        border-color: rgba(108, 99, 255, 0.3);
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Input Group Labels */
    .input-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.6);
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Sub-section Divider */
    .sub-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 20px 0;
    }
    
    /* Model Selection Card */
    .model-selector {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.15) 0%, rgba(255, 101, 132, 0.1) 100%);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 16px;
        padding: 20px 25px;
        margin-bottom: 30px;
    }
    
    /* Submit Button Enhancement */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF 0%, #FF6584 100%) !important;
        font-size: 1.15rem !important;
        padding: 1rem 2.5rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }
    
    /* Column Equal Heights */
    [data-testid="column"] > div {
        height: 100%;
    }
    
    /* Time Input Alignment */
    .stTimeInput {
        margin-top: 0 !important;
    }
    
    /* Number Input Cleanup */
    .stNumberInput > div {
        margin-bottom: 0 !important;
    }
    
    /* Equal Height Containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        height: 100%;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
MODELS_DIR = "project_models"
MODEL_FILES = {
    "xgboost": "xgboost_bsq_model.pkl",
    "lightgbm": "lightgbm_bsq_model.pkl",
    "random_forest": "random_forest_bsq_model.pkl",
    "mlp": "mlp_bsq_model.pth",
    "mlp_scaler": "mlp_scaler.pkl"
}

# --- CORRECT MLP ARCHITECTURE (matches trained model) ---
# The actual trained model uses 46 input features with BatchNorm layers
class SleepQualityMLP(nn.Module):
    """
    MLP architecture matching the trained model:
    Linear(46→256) → BatchNorm → LeakyReLU → Dropout →
    Linear(256→128) → BatchNorm → LeakyReLU → Dropout →
    Linear(128→64) → BatchNorm → LeakyReLU → Dropout →
    Linear(64→32) → BatchNorm → LeakyReLU → Dropout →
    Linear(32→1) → Sigmoid
    """
    def __init__(self, input_dim=46):
        super(SleepQualityMLP, self).__init__()
        self.network = nn.Sequential(
            # Layer 0-3: Input → 256
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            # Layer 4-7: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            # Layer 8-11: 128 → 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            # Layer 12-15: 64 → 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            # Layer 16: 32 → 1
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# MLP Feature List (46 features in exact order from scaler)
MLP_FEATURES = [
    'Age', 'BMI', 'Actual_sleep_hours', 'Sleep_latency', 'Chronotype_MEQ',
    'ScoreMEQ', 'Eating_window_workday', 'Eating_window_freeday',
    'Bedtime_min', 'Wake_up_time_min', 'Last_eating_workday_min', 'Last_eating_freeday_min',
    'Gender_1', 'Gender_2', 'Industry_1', 'Industry_2',
    'Part-time_0.0', 'Part-time_1.0', 'Smoking/Vaping_0.0', 'Smoking/Vaping_1.0',
    'Shift_Rotation_1.0', 'Shift_Rotation_2.0', 'Shift_Rotation_3.0', 'Shift_Rotation_4.0',
    'Chronotype_Shift_1.0', 'Chronotype_Shift_2.0', 'Chronotype_Shift_3.0',
    'Breakfast_skipping_0.0', 'Breakfast_skipping_1.0', 'Breakfast_skipping_2.0',
    'Breakfast_skipping_3.0', 'Breakfast_skipping_4.0', 'Breakfast_skipping_5.0',
    'Breakfast_skipping_6.0', 'Breakfast_skipping_7.0',
    'Largest_mealtime_0', 'Largest_mealtime_1', 'Largest_mealtime_2', 'Largest_mealtime_3',
    'Largest_mealtime_4', 'Largest_mealtime_1,2', 'Largest_mealtime_1,2,3',
    'Largest_mealtime_1,2,3,', 'Largest_mealtime_1,2,3,4', 'Largest_mealtime_1,3', 'Largest_mealtime_2,3'
]

# --- MODEL LOADER ---
@st.cache_resource
def load_ensemble():
    models = {}
    
    # 1. XGBoost
    try:
        path = os.path.join(MODELS_DIR, MODEL_FILES["xgboost"])
        if os.path.exists(path):
            models["xgboost"] = joblib.load(path)
    except Exception as e:
        st.error(f"⚠️ Failed to load XGBoost: {e}")

    # 2. LightGBM
    try:
        path = os.path.join(MODELS_DIR, MODEL_FILES["lightgbm"])
        if os.path.exists(path):
            models["lightgbm"] = joblib.load(path)
    except Exception as e:
        st.warning(f"⚠️ Failed to load LightGBM: {e}")

    # 3. Random Forest
    try:
        path = os.path.join(MODELS_DIR, MODEL_FILES["random_forest"])
        if os.path.exists(path):
            models["random_forest"] = joblib.load(path)
    except Exception as e:
        st.warning(f"⚠️ Failed to load Random Forest: {e}")

    # 4. MLP (PyTorch) - Using correct architecture
    try:
        model_path = os.path.join(MODELS_DIR, MODEL_FILES["mlp"])
        scaler_path = os.path.join(MODELS_DIR, MODEL_FILES["mlp_scaler"])
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            models["mlp_scaler"] = joblib.load(scaler_path)
            
            # Initialize with CORRECT architecture (46 input features)
            mlp = SleepQualityMLP(input_dim=46)
            device = torch.device('cpu')
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            mlp.load_state_dict(state_dict)
            mlp.eval()
            models["mlp"] = mlp
    except Exception as e:
        st.warning(f"⚠️ Failed to load MLP: {e}")

    return models

# Initiate loading
models = load_ensemble()

# --- PREPROCESSING LOGIC ---
CANONICAL_FEATURES = [
    'Actual_sleep_hours', 'Age', 'BMI', 'Bedtime', 
    'Breakfast_skipping_0', 'Breakfast_skipping_1', 'Breakfast_skipping_2', 'Breakfast_skipping_3', 
    'Breakfast_skipping_4', 'Breakfast_skipping_5', 'Breakfast_skipping_6', 'Breakfast_skipping_7', 
    'Chronotype_MEQ', 
    'Chronotype_Shift_1', 'Chronotype_Shift_2', 'Chronotype_Shift_3', 
    'Eating_window_freeday', 'Eating_window_workday', 
    'Gender_1', 'Gender_2', 
    'Industry_2', 
    'Largest_mealtime_2', 'Largest_mealtime_3', 
    'Last_eating_freeday', 'Last_eating_workday', 
    'Part-time_0', 
    'ScoreMEQ', 
    'Shift_Rotation_2', 'Shift_Rotation_3', 
    'Sleep_latency', 
    'Smoking/Vaping_0', 'Smoking/Vaping_1', 
    'Wake_up_time'
]

def time_to_min(time_val):
    """Converts HH:MM string to minutes since midnight."""
    if not time_val:
        return 0
    try:
        # Check if it's already a time object (from st.time_input)
        if hasattr(time_val, 'hour'):
            return time_val.hour * 60 + time_val.minute
        # Else assume string HH:MM
        h, m = map(int, str(time_val).split(':'))
        return h * 60 + m
    except:
        return 0

def preprocess_inputs(inputs):
    # 1. Base dictionary of 0.0s
    encoded = {col: 0.0 for col in CANONICAL_FEATURES}
    
    # 2. Fill Numerics
    encoded['Age'] = float(inputs.get('Age', 0))
    encoded['BMI'] = float(inputs.get('BMI', 0))
    encoded['Actual_sleep_hours'] = float(inputs.get('Actual_sleep_hours', 0))
    encoded['Sleep_latency'] = float(inputs.get('Sleep_latency', 0))
    encoded['Chronotype_MEQ'] = float(inputs.get('Chronotype_MEQ', 0))
    encoded['ScoreMEQ'] = float(inputs.get('ScoreMEQ', 0))
    encoded['Eating_window_workday'] = float(inputs.get('Eating_window_workday', 0))
    encoded['Eating_window_freeday'] = float(inputs.get('Eating_window_freeday', 0))
    
    # 3. Time Conversions
    encoded['Bedtime'] = float(time_to_min(inputs.get('Bedtime')))
    encoded['Wake_up_time'] = float(time_to_min(inputs.get('Wake_up_time')))
    encoded['Last_eating_workday'] = float(time_to_min(inputs.get('Last_eating_workday')))
    encoded['Last_eating_freeday'] = float(time_to_min(inputs.get('Last_eating_freeday')))
    
    # 4. Categorical Mapping
    gender_map = {'Male': 'Gender_1', 'Female': 'Gender_2'}
    if inputs['Gender'] in gender_map:
        key = gender_map[inputs['Gender']]
        if key in encoded: encoded[key] = 1.0

    if inputs['Industry'] == 'Non-Healthcare Workers': 
        encoded['Industry_2'] = 1.0 
        
    if inputs['Part-time'] == 'No':
        encoded['Part-time_0'] = 1.0
        
    if inputs['Smoking/Vaping'] == 'No':
        encoded['Smoking/Vaping_0'] = 1.0
    elif inputs['Smoking/Vaping'] == 'Yes':
        encoded['Smoking/Vaping_1'] = 1.0
        
    bf_val = inputs['Breakfast_skipping']
    bf_col = f"Breakfast_skipping_{bf_val}"
    if bf_col in encoded: encoded[bf_col] = 1.0
    
    
    lm_val = inputs['Largest_mealtime']
    lm_col = f"Largest_mealtime_{lm_val}"
    if lm_col in encoded: encoded[lm_col] = 1.0
    
    c_shift = inputs['Chronotype_Shift']
    c_col = f"Chronotype_Shift_{c_shift}"
    if c_col in encoded: encoded[c_col] = 1.0
    
    s_rot = inputs['Shift_Rotation']
    s_col = f"Shift_Rotation_{s_rot}"
    if s_col in encoded: encoded[s_col] = 1.0

    return pd.DataFrame([encoded], columns=CANONICAL_FEATURES)

def preprocess_mlp_inputs(inputs):
    """
    Specialized preprocessing for the 46-feature MLP model.
    Handles strict one-hot encoding requirements and correct feature scaling.
    """
    # Create base dictionary with 0.0 for all 46 features
    encoded = {col: 0.0 for col in MLP_FEATURES}
    
    # 1. Numerics
    encoded['Age'] = float(inputs.get('Age', 0))
    encoded['BMI'] = float(inputs.get('BMI', 0))
    encoded['Actual_sleep_hours'] = float(inputs.get('Actual_sleep_hours', 0))
    encoded['Sleep_latency'] = float(inputs.get('Sleep_latency', 0))
    encoded['Chronotype_MEQ'] = float(inputs.get('Chronotype_MEQ', 0))
    encoded['ScoreMEQ'] = float(inputs.get('ScoreMEQ', 0))
    encoded['Eating_window_workday'] = float(inputs.get('Eating_window_workday', 0))
    encoded['Eating_window_freeday'] = float(inputs.get('Eating_window_freeday', 0))
    
    # Time conversions (HH:MM -> Minutes)
    encoded['Bedtime_min'] = float(time_to_min(inputs.get('Bedtime')))
    encoded['Wake_up_time_min'] = float(time_to_min(inputs.get('Wake_up_time')))
    encoded['Last_eating_workday_min'] = float(time_to_min(inputs.get('Last_eating_workday')))
    encoded['Last_eating_freeday_min'] = float(time_to_min(inputs.get('Last_eating_freeday')))

    # 2. Categorical - One Hot Encoding (Strict Mapping)
    
    # Gender (1=Male, 2=Female)
    if inputs['Gender'] == 'Male': encoded['Gender_1'] = 1.0
    elif inputs['Gender'] == 'Female': encoded['Gender_2'] = 1.0
    
    # Industry (1=Healthcare, 2=Non-Healthcare)
    if inputs['Industry'] == 'Healthcare Workers': encoded['Industry_1'] = 1.0
    elif inputs['Industry'] == 'Non-Healthcare Workers': encoded['Industry_2'] = 1.0

    # Part-time (0.0=No, 1.0=Yes)
    if inputs['Part-time'] == 'No': encoded['Part-time_0.0'] = 1.0
    elif inputs['Part-time'] == 'Yes': encoded['Part-time_1.0'] = 1.0
    
    # Smoking (0.0=No, 1.0=Yes)
    if inputs['Smoking/Vaping'] == 'No': encoded['Smoking/Vaping_0.0'] = 1.0
    elif inputs['Smoking/Vaping'] == 'Yes': encoded['Smoking/Vaping_1.0'] = 1.0
    
    # Shift Rotation (1.0, 2.0, 3.0)
    rot_val = float(inputs['Shift_Rotation'])
    rot_col = f"Shift_Rotation_{rot_val}"
    if rot_col in encoded: encoded[rot_col] = 1.0
    
    # Chronotype Shift (1.0, 2.0, 3.0)
    c_shift = float(inputs['Chronotype_Shift'])
    c_col = f"Chronotype_Shift_{c_shift}"
    if c_col in encoded: encoded[c_col] = 1.0
    
    # Breakfast Skipping (0.0 - 7.0)
    bf_val = float(inputs['Breakfast_skipping'])
    bf_col = f"Breakfast_skipping_{bf_val}"
    if bf_col in encoded: encoded[bf_col] = 1.0
    
    # Largest Meal (0, 1, 2, 3...)
    # UI gives 1 (Breakfast), 2 (Lunch), 3 (Dinner)
    lm_val = inputs['Largest_mealtime']
    lm_col = f"Largest_mealtime_{lm_val}"
    if lm_col in encoded: encoded[lm_col] = 1.0
    
    return pd.DataFrame([encoded], columns=MLP_FEATURES)

# =====================================================
# UI LAYOUT
# =====================================================

# Page Header
st.markdown('<div class="main-title">🌙 AI Sleep Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Sleep Quality Prediction powered by Deep Learning Ensemble</div>', unsafe_allow_html=True)

# Form
with st.form("prediction_form"):
    
    # MODEL SELECTION - Styled Card
    st.markdown('<div class="section-header">🤖 Select AI Model</div>', unsafe_allow_html=True)
    
    model_cols = st.columns([2, 3])
    with model_cols[0]:
        model_choice = st.selectbox(
            "Prediction Engine", 
            ["Ensemble (Recommended)", "XGBoost", "LightGBM", "Random Forest", "Neural Network (MLP)"],
            help="Select a specific algorithm or use Ensemble to combine all for best accuracy.",
            label_visibility="collapsed"
        )
    with model_cols[1]:
        st.markdown("""
        <div style="padding: 10px 15px; background: rgba(108, 99, 255, 0.1); border-radius: 10px; font-size: 0.9rem; color: rgba(255,255,255,0.7);">
            💡 <strong>Tip:</strong> Ensemble mode combines XGBoost, LightGBM, Random Forest & Neural Network for highest accuracy.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== MAIN INPUT SECTIONS ==========
    col1, col2, col3 = st.columns(3, gap="medium")
    
    # --- COLUMN 1: PERSONAL PROFILE ---
    with col1:
        with st.container(border=True):
            st.markdown('<div class="section-header">👤 Personal Profile</div>', unsafe_allow_html=True)
            
            # Age
            age = st.number_input("Age", 18, 100, 30, help="Your current age in years")
            
            # Gender & BMI Row
            g_col, b_col = st.columns(2)
            with g_col:
                gender = st.selectbox("Gender", ["Male", "Female"])
            with b_col:
                bmi = st.number_input("BMI", 10.0, 50.0, 24.0, format="%.1f", help="Body Mass Index")
            
            # Lifestyle
            st.markdown('<div class="sub-divider"></div>', unsafe_allow_html=True)
            smoking = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker/Vaping"], help="Current smoking or vaping habits")
            
        # Work Section
        with st.container(border=True):
            st.markdown('<div class="section-header">🏢 Work Environment</div>', unsafe_allow_html=True)
            
            industry = st.selectbox("Industry", ["Healthcare Workers", "Non-Healthcare Workers"])
            
            i_col, p_col = st.columns(2)
            with i_col:
                part_time = st.selectbox("Part-time?", ["No", "Yes"])
            with p_col:
                shift_type = st.selectbox("Shift Type", ["Fixed (Day)", "Rotating", "Irregular"])

    # --- COLUMN 2: SLEEP PATTERNS ---
    with col2:
        with st.container(border=True):
            st.markdown('<div class="section-header">💤 Sleep Patterns</div>', unsafe_allow_html=True)
            
            # Primary Sleep Metrics
            act_sleep = st.number_input("Actual Sleep Hours", 0.0, 24.0, 7.0, step=0.5, format="%.1f")
            latency = st.number_input("Sleep Latency (mins)", 0, 300, 15, help="Time taken to fall asleep")
            
            # Bedtime / Waketime
            st.markdown('<div class="sub-divider"></div>', unsafe_allow_html=True)
            time_cols = st.columns(2)
            with time_cols[0]:
                bedtime = st.time_input("🌙 Bedtime", value=datetime.strptime("23:00", "%H:%M").time())
            with time_cols[1]:
                wake_time = st.time_input("☀️ Wake Up", value=datetime.strptime("07:00", "%H:%M").time())
            
        # Chronotype Section
        with st.container(border=True):
            st.markdown('<div class="section-header">⏰ Chronotype</div>', unsafe_allow_html=True)
            
            meq_cols = st.columns(2)
            with meq_cols[0]:
                meq_score = st.number_input("MEQ Score", 0, 100, 50, help="Morningness-Eveningness Questionnaire (0-100)")
            with meq_cols[1]:
                chronotype_meq = st.number_input("Chronotype MEQ", 0.0, 10.0, 4.0, step=0.1)
            
            chrono_shift = st.selectbox("Circadian Preference", ["Morning", "Intermediate", "Evening"])

    # --- COLUMN 3: NUTRITION ---
    with col3:
        with st.container(border=True):
            st.markdown('<div class="section-header">🍽️ Nutrition & Diet</div>', unsafe_allow_html=True)
            
            meal_time = st.selectbox("Largest Meal of Day", ["Breakfast", "Lunch", "Dinner"])
            breakfast = st.slider("Breakfast Skipping (days/week)", 0, 7, 0)
            
            # Workday Eating
            st.markdown('<div class="sub-divider"></div>', unsafe_allow_html=True)
            st.markdown("**📅 Workday Eating**")
            work_cols = st.columns(2)
            with work_cols[0]:
                eat_win_work = st.number_input("Window (hrs)", 0, 24, 12, key="eww", help="Eating window duration")
            with work_cols[1]:
                last_eat_work = st.time_input("Last Meal", value=datetime.strptime("20:00", "%H:%M").time(), key="lew")
            
            # Freeday Eating
            st.markdown("**🌴 Freeday Eating**")
            free_cols = st.columns(2)
            with free_cols[0]:
                eat_win_free = st.number_input("Window (hrs)", 0, 24, 12, key="ewf", help="Eating window duration")
            with free_cols[1]:
                last_eat_free = st.time_input("Last Meal", value=datetime.strptime("21:00", "%H:%M").time(), key="lef")
    
    # ========== SUBMIT BUTTON ==========
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        submit = st.form_submit_button("✨  ANALYZE SLEEP QUALITY  ✨", use_container_width=True)

# =====================================================
# PREDICTION LOGIC
# =====================================================
if submit:
    # Mapping UI to raw inputs
    chrono_shift_map = {"Morning": 1, "Evening": 2, "Intermediate": 3}
    rotation_map = {"Fixed (Day)": 1, "Rotating": 2, "Irregular": 3}
    meal_map = {"Breakfast": 1, "Lunch": 2, "Dinner": 3}
    smoking_map = "Yes" if smoking == "Smoker/Vaping" else "No"

    raw_inputs = {
        "Age": age,
        "BMI": bmi,
        "Gender": gender,
        "Smoking/Vaping": smoking_map,
        "Actual_sleep_hours": act_sleep,
        "Sleep_latency": latency,
        "Bedtime": bedtime,
        "Wake_up_time": wake_time,
        "ScoreMEQ": meq_score,
        "Chronotype_MEQ": chronotype_meq,
        "Industry": industry,
        "Part-time": part_time,
        "Shift_Rotation": rotation_map[shift_type],
        "Chronotype_Shift": chrono_shift_map[chrono_shift],
        "Breakfast_skipping": breakfast,
        "Largest_mealtime": meal_map[meal_time],
        "Eating_window_workday": eat_win_work,
        "Last_eating_workday": last_eat_work,
        "Eating_window_freeday": eat_win_free,
        "Last_eating_freeday": last_eat_free
    }

    # Process
    X = preprocess_inputs(raw_inputs)
    
    # Inference
    preds = []
    details = {}
    
    # XGB
    if "xgboost" in models:
        try:
            p = models["xgboost"].predict(X)[0]
            preds.append(p)
            details["XGBoost"] = float(p)
        except Exception as e:
            details["XGBoost"] = f"Error: {e}"

    # LGBM
    if "lightgbm" in models:
        try:
            p = models["lightgbm"].predict(X)[0]
            preds.append(p)
            details["LightGBM"] = float(p)
        except: pass

    # RF
    if "random_forest" in models:
        try:
            p = models["random_forest"].predict(X)[0]
            preds.append(p)
            details["Random Forest"] = float(p)
        except: pass
        
    # MLP
    if "mlp" in models and "mlp_scaler" in models:
        try:
            # 1. Use Specialized Preprocessing (46 Features)
            X_mlp = preprocess_mlp_inputs(raw_inputs)
            
            # 2. Scale
            X_scaled = models["mlp_scaler"].transform(X_mlp)
            
            # 3. Predict
            X_t = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                p = models["mlp"](X_t).item()
            preds.append(p)
            details["MLP"] = float(p)
        except Exception as e:
            details["MLP"] = f"Error: {e}"

    # Result Calculation
    if preds:
        ensemble_score = np.mean(preds)
        
        primary_score = ensemble_score
        primary_source = "Ensemble"
        
        if model_choice == "XGBoost" and "XGBoost" in details:
            primary_score = details["XGBoost"]
            primary_source = "XGBoost"
        elif model_choice == "LightGBM" and "LightGBM" in details:
            primary_score = details["LightGBM"]
            primary_source = "LightGBM"
        elif model_choice == "Random Forest" and "Random Forest" in details:
            primary_score = details["Random Forest"]
            primary_source = "Random Forest"
        elif model_choice == "Neural Network (MLP)" and "MLP" in details:
            primary_score = details["MLP"]
            primary_source = "Neural Network"
            
        # Label Logic
        if primary_score < 0.50:
            label = "POOR SLEEP QUALITY"
            color = "#FF6B6B"
            icon = "⚠️"
            emoji = "😔"
        elif primary_score < 0.75:
            label = "MODERATE SLEEP QUALITY"
            color = "#FFB347"
            icon = "⚖️"
            emoji = "😐"
        else:
            label = "GOOD SLEEP QUALITY"
            color = "#1DD1A1"
            icon = "✨"
            emoji = "😊"
        
        # Display Result
        st.markdown("<br>", unsafe_allow_html=True)
        
        result_col1, result_col2, result_col3 = st.columns([1, 3, 1])
        with result_col2:
            st.markdown(f"""
            <div class="result-container" style="border: 2px solid {color}; box-shadow: 0 0 40px {color}40;">
                <div style="font-size: 3rem; margin-bottom: 10px;">{emoji}</div>
                <p style="color: rgba(255,255,255,0.6); letter-spacing: 2px; text-transform: uppercase; font-size: 0.9rem; margin-bottom: 5px;">
                    {primary_source} Prediction
                </p>
                <div class="score-display">{primary_score:.3f}</div>
                <div class="badge" style="background: linear-gradient(135deg, {color} 0%, {color}CC 100%); color: white; box-shadow: 0 8px 25px {color}60;">
                    {icon} {label}
                </div>
                <p style="margin-top: 25px; color: rgba(255,255,255,0.5); font-size: 0.85rem;">
                    BSQI Score Range: 0.0 (Poor) → 1.0 (Excellent)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Ensemble Comparison
        if model_choice != "Ensemble (Recommended)":
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"💡 **Ensemble Benchmark:** The combined average of all models is **{ensemble_score:.3f}**. "
                    f"Comparing your selected model ({primary_score:.3f}) with the consensus helps identify outliers.")
        
        # Model Breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 View AI Model Breakdown", expanded=False):
            model_cols = st.columns(len(details))
            for idx, (model_name, score) in enumerate(details.items()):
                with model_cols[idx]:
                    if isinstance(score, float):
                        score_color = "#1DD1A1" if score >= 0.75 else ("#FFB347" if score >= 0.5 else "#FF6B6B")
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.03); border-radius: 12px;">
                            <div style="font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-bottom: 8px;">{model_name}</div>
                            <div style="font-size: 1.8rem; font-weight: 700; color: {score_color};">{score:.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; background: rgba(255,100,100,0.1); border-radius: 12px;">
                            <div style="font-size: 0.85rem; color: rgba(255,255,255,0.6);">{model_name}</div>
                            <div style="font-size: 0.9rem; color: #FF6B6B;">Error</div>
                        </div>
                        """, unsafe_allow_html=True)
            
    else:
        st.error("❌ No models available for prediction. Please check the model files.")

# Footer
st.markdown("""
<div style="text-align: center; padding: 40px 0 20px 0; color: rgba(255,255,255,0.4); font-size: 0.85rem;">
    AI Sleep Lab • Powered by Deep Learning Ensemble
</div>
""", unsafe_allow_html=True)
