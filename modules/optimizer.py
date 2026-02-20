import pandas as pd
import numpy as np
import random
import streamlit as st
from modules import ml_model

def optimize_schedule(current_profile, model, encoders, feature_names, n_iter=500):
    """
    Finds a better lifestyle configuration for the user.
    
    Args:
        current_profile (dict): Uses current inputs.
        model: Trained ML model.
        encoders: LabelEncoders.
        feature_names: List of features expected by model.
        n_iter (int): Number of random schedules to test.
        
    Returns:
        best_profile (dict): Optimized profile.
        improvement (float): Reduction in bad sleep probability.
        original_prob (float): Original probability.
    """
    
    # 1. Baseline Prediction
    original_prob, _ = ml_model.predict_single(model, current_profile, encoders, feature_names)
    
    if original_prob < 0.2:
        # If already good, don't optimize much, just return
        return current_profile, 0.0, original_prob

    # 2. Define Constraints (What can we change?)
    # Actionable: Bedtime, Wake Time, Eating Window, Sleep Hours, Breakfast
    # Fixed: Age, Gender, Shift, Chronotype, Industry, etc.
    
    best_profile = current_profile.copy()
    min_prob = original_prob
    
    # Bounds for randomization
    # Bedtime: 21:00 (21.0) to 03:00 (27.0) 
    # Wake time: 05:00 (5.0) to 12:00 (12.0)
    # Sleep hours: 6.0 to 9.0 (Healthy range)
    # Eating window: 8.0 to 12.0 (Restricted feeding is often better)
    
    candidates = []
    
    for _ in range(n_iter):
        candidate = current_profile.copy()
        
        # --- Randomize Actionable Features ---
        
        # 1. Sleep Duration (Aim for healthy 7-9h)
        new_sleep = random.uniform(6.5, 9.0)
        candidate['Actual_sleep_hours'] = new_sleep
        candidate['Actual_sleep_hours_decimal'] = new_sleep # if used
        
        # 2. Bedtime (Shift it slightly or drastically)
        # Convert HH:MM strings to decimals for math if needed, but model handles it
        # Let's assume input is HH:MM string, so we generate reasonable strings
        
        # Heuristic: Early to bed is generally better?
        # Generate random bedtime between 20:00 and 02:00
        # Represent as float 20.0 to 26.0 (02:00)
        bedtime_float = random.uniform(20.0, 26.0) 
        if bedtime_float >= 24.0:
            bed_h = int(bedtime_float - 24)
            bed_m = int((bedtime_float - int(bedtime_float)) * 60)
            candidate['Bedtime'] = f"{bed_h:02d}:{bed_m:02d}"
        else:
            bed_h = int(bedtime_float)
            bed_m = int((bedtime_float - int(bedtime_float)) * 60)
            candidate['Bedtime'] = f"{bed_h:02d}:{bed_m:02d}"
            
        # 3. Wake Time (Derived from Bedtime + Sleep)
        wake_float = bedtime_float + new_sleep
        if wake_float >= 24.0:
            wake_float -= 24.0
        if wake_float >= 24.0: # Next day overlap
             wake_float -= 24.0
             
        wake_h = int(wake_float)
        wake_m = int((wake_float - int(wake_float)) * 60)
        candidate['Wake_up_time'] = f"{wake_h:02d}:{wake_m:02d}"
        
        # 4. Eating Window (Time-Restricted Eating)
        # Try reducing window to 8-12 hours
        candidate['Eating_window_workday'] = random.uniform(8.0, 12.0)
        
        # 5. Breakfast (Try both, bias towards Yes)
        if random.random() > 0.3:
            candidate['Breakfast_skipping'] = "No"
        else:
            candidate['Breakfast_skipping'] = "Yes"
            
        # 6. Sleep Latency (Can't directly control, but better hygiene helps)
        # Assume optimization implies better hygiene -> lower latency
        candidate['Sleep_latency'] = max(5, current_profile['Sleep_latency'] * 0.8) 
        
        candidates.append(candidate)
        
    # 3. Batch Predict (or Loop if batch not supported by singleton predict)
    # Since predict_single is designed for UI, we'll loop. 
    # For speed, we could vectorize this, but N=500 is fast enough for inference.
    
    for cand in candidates:
        prob, _ = ml_model.predict_single(model, cand, encoders, feature_names)
        
        if prob < min_prob:
            min_prob = prob
            best_profile = cand
            
    improvement = (original_prob - min_prob) / original_prob if original_prob > 0 else 0
    
    return best_profile, min_prob, original_prob
