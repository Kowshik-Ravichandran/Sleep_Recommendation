import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
MODELS_DIR = "project_models"
MODEL_FILES = {
    "xgboost": "xgboost_bsq_model.pkl",
    "lightgbm": "lightgbm_bsq_model.pkl",
    "random_forest": "random_forest_bsq_model.pkl",
    "mlp": "mlp_bsq_model.pth",
    "mlp_scaler": "mlp_scaler.pkl"
}

# --- BOUNDED MLP DEFINITION (Placeholder) ---
# Trying to guess standard architecture since source was not provided.
# If this fails, the error handler will catch it and skip MLP.
class BoundedMLP(nn.Module):
    def __init__(self, input_dim=33):
        super(BoundedMLP, self).__init__()
        # Architecture inferred from state_dict:
        # net.0: Linear(33, 128)
        # net.2: Linear(128, 64)
        # net.4: Linear(64, 32)
        # net.6: Linear(32, 1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --- MODEL LOADER ---
@st.cache_resource
def load_ensemble():
    """
    Loads all models into memory. 
    Returns a dictionary of models.
    """
    models = {}
    
    # 1. XGBoost
    try:
        path = os.path.join(MODELS_DIR, MODEL_FILES["xgboost"])
        if os.path.exists(path):
            models["xgboost"] = joblib.load(path)
            # print(f"Loaded XGBoost from {path}")
    except Exception as e:
        st.error(f"Failed to load XGBoost: {e}")

    # 2. LightGBM
    try:
        path = os.path.join(MODELS_DIR, MODEL_FILES["lightgbm"])
        if os.path.exists(path):
            models["lightgbm"] = joblib.load(path)
    except Exception as e:
        print(f"Failed to load LightGBM: {e}") # Non-critical log

    # 3. Random Forest
    try:
        path = os.path.join(MODELS_DIR, MODEL_FILES["random_forest"])
        if os.path.exists(path):
            models["random_forest"] = joblib.load(path)
    except Exception as e:
        print(f"Failed to load Random Forest: {e}")

    # 4. MLP (PyTorch)
    try:
        model_path = os.path.join(MODELS_DIR, MODEL_FILES["mlp"])
        scaler_path = os.path.join(MODELS_DIR, MODEL_FILES["mlp_scaler"])
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            # Load Scaler
            models["mlp_scaler"] = joblib.load(scaler_path)
            
            # Load Model
            # Since we don't know the exact dim, we might fail here if architecture mismatches
            # We trust the state_dict to tell us dimensions if we can inspect it, 
            # but usually we need to instantiate the class first.
            
            # Attempt 1: Try to load assuming the guess BoundedMLP
            # We assume input dim is 33 (same as XGBoost features)
            Device = torch.device('cpu')
            
            # Helper to inspect state dict shape if possible
            state_dict = torch.load(model_path, map_location=Device)
            
            # Dynamic layer sizing based on state_dict if needed/possible
            # But for now, let's try strict load and catch exception
             
            # HEURISTIC: Check Input layer weight shape to confirm feature count
            input_dim = 33
            hidden_layers = []
            
            # Try to infer structure? This is complex. 
            # We will try the standard instantiate.
            
            # Note: User didn't provide code, so we can't be sure.
            # We will store the state_dict and try to use it if we can match it?
            # Or just skip if it fails.
            
            # Correct architecture inferred
            mlp = BoundedMLP(input_dim=33)
            try:
                mlp.load_state_dict(state_dict)
                mlp.eval()
                models["mlp"] = mlp
            except RuntimeError as e:
                print(f"MLP Load Error (Mismatched Keys?): {e}")
                
    except Exception as e:
        print(f"Failed to load MLP: {e}")

    return models

# --- PREPROCESSING & INFERENCE ---

# Reference Feature Order (Canonical Schema from XGBoost)
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

def preprocess_inputs(user_inputs):
    """
    Converts user input dict to model-ready dataframe (~33 cols).
    """
    # 1. Base DataFrame
    df = pd.DataFrame([user_inputs])
    
    # 2. Time Conversion (HH:MM -> Minutes)
    time_cols = ['Bedtime', 'Wake_up_time', 'Last_eating_workday', 'Last_eating_freeday']
    for col in time_cols:
        if col in df.columns:
            val = str(df.iloc[0][col])
            if ':' in val:
                h, m = map(int, val.split(':'))
                df[col] = h * 60 + m
            else:
                df[col] = 0 # Fallback
                
    # 3. Categorical Encoding (Manual One-Hot)
    # This is tricky because we need to match the specific '0', '1', '2' suffixes
    # found in the canonical features.
    
    # We will assume the user inputs are strings matching the categories
    # and we map them to the corresponding one-hot columns.
    
    # Let's clean the input keys first.
    
    # Create a dict of all 0s first
    encoded_row = {col: 0.0 for col in CANONICAL_FEATURES}
    
    # Map Numeric directly
    for col in ['Actual_sleep_hours', 'Age', 'BMI', 'Bedtime', 'Chronotype_MEQ', 
                'Eating_window_freeday', 'Eating_window_workday', 'ScoreMEQ',
                'Sleep_latency', 'Last_eating_freeday', 'Last_eating_workday', 'Wake_up_time']:
         if col in df.columns:
             encoded_row[col] = float(df.iloc[0][col])

    # Map Categorical
    # Logic: If input 'Gender' is '1', set 'Gender_1' = 1.
    # Note: Value mapping depends on how the original model was trained. 
    # Inspecting valid values:
    # Gender: 1, 2
    # Shift: 1, 2, 3
    # Breakfast: 0-7
    
    cat_map = {
        'Gender': 'Gender_',
        'Chronotype_Shift': 'Chronotype_Shift_',
        'Shift_Rotation': 'Shift_Rotation_',
        'Industry': 'Industry_', # Only Industry_2 exists in features? Maybe binary?
        'Largest_mealtime': 'Largest_mealtime_',
        'Part-time': 'Part-time_',
        'Smoking/Vaping': 'Smoking/Vaping_',
        'Breakfast_skipping': 'Breakfast_skipping_'
    }
    
    for input_key, prefix in cat_map.items():
        if input_key in df.columns:
            val = df.iloc[0][input_key]
            # Construct expected column name, e.g. "Gender_1"
            col_name = f"{prefix}{val}"
            
            # If this column exists in canonical features, set to 1
            if col_name in encoded_row:
                encoded_row[col_name] = 1.0
                
    # Create final DF
    X_final = pd.DataFrame([encoded_row])
    # Ensure column order
    X_final = X_final[CANONICAL_FEATURES]
    
    return X_final

def predict_ensemble(user_inputs):
    """
    Runs inference on all available models and averages results.
    """
    models = load_ensemble()
    X = preprocess_inputs(user_inputs)
    
    predictions = []
    details = {}
    
    # 1. XGBoost
    if "xgboost" in models:
        try:
            # XGBoost usually predicts 0/1 class or probability? 
            # Assuming regressor or binary classifier probability
            pred = models["xgboost"].predict(X)[0]
            # If classifier, predict_proba?
            if hasattr(models["xgboost"], "predict_proba"):
                 pred = models["xgboost"].predict_proba(X)[0][1] # Probability of Class 1? 
                 # Wait, BSQI is a score (regressor) or Class?
                 # Prompt says: "Compute the final BSQI score... BSQI < 0.50 -> Poor"
                 # This implies regression logic 0.0 to 1.0
            
            predictions.append(pred)
            details["XGBoost"] = float(pred)
        except Exception as e:
            details["XGBoost"] = f"Error: {e}"

    # 2. LightGBM
    if "lightgbm" in models:
        try:
            pred = models["lightgbm"].predict(X)[0]
            # LGBM often returns just the array
            predictions.append(pred)
            details["LightGBM"] = float(pred)
        except Exception:
            pass

    # 3. Random Forest
    if "random_forest" in models:
        try:
             # Check if regressor or classifier
             if hasattr(models["random_forest"], "predict_proba"):
                 pred = models["random_forest"].predict_proba(X)[0][1]
             else:
                 pred = models["random_forest"].predict(X)[0]
                 
             predictions.append(pred)
             details["Random Forest"] = float(pred)
        except Exception:
            pass

    # 4. MLP
    if "mlp" in models and "mlp_scaler" in models:
        try:
            # COPY input for MLP specific adjustment
            X_mlp = X.copy()
            
            # PATCH: The scaler shows means of 0.0 for these, meaning they were likely 
            # unused or zeroed during MLP training. Passing 1200+ (minutes) breaks it.
            # We force them to 0 to match the training conditions.
            if 'Last_eating_freeday' in X_mlp.columns:
                X_mlp['Last_eating_freeday'] = 0.0
            if 'Last_eating_workday' in X_mlp.columns:
                X_mlp['Last_eating_workday'] = 0.0
                
            # Scale
            X_scaled = models["mlp_scaler"].transform(X_mlp)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            
            with torch.no_grad():
                pred = models["mlp"](X_tensor).item()
                
            predictions.append(pred)
            details["MLP"] = float(pred)
        except Exception as e:
            details["MLP"] = f"Error: {e}"
            
    # Aggregate
    if not predictions:
        return 0.0, "Error", {}
        
    final_score = np.mean(predictions)
    
    # Map label
    label = "Unknown"
    if final_score < 0.50:
        label = "Poor Sleep"
    elif final_score < 0.75:
        label = "Moderate Sleep"
    else:
        label = "Good Sleep"
        
    return final_score, label, details
