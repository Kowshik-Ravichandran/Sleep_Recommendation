import pandas as pd
import numpy as np

def generate_insights(df):
    """
    Analyzes the dataframe to produce natural language insights.
    Returns specific findings for Shifts, Chronotypes, and Nutrition.
    """
    insights = {
        "correlations": [],
        "worst_shift": "",
        "risk_chronotype": "",
        "nutrition": ""
    }
    
    # helper for creating numeric
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 1. Detect Correlations
    # Focus on correlation with Total_scorePSQI or Actual_sleep_hours
    if 'Total_scorePSQI' in numeric_df.columns:
        corr = numeric_df.corr()['Total_scorePSQI'].sort_values(ascending=False)
        # Top positive correlation (worse sleep)
        top_corr = corr.drop('Total_scorePSQI').head(1)
        if not top_corr.empty:
            insights['correlations'].append(f"Strongest predictor of poor sleep (High PSQI) is **{top_corr.index[0]}** (Correlation: {top_corr.values[0]:.2f})")
    
    # 2. Worst Shift
    if 'Shift_Rotation' in df.columns and 'Total_scorePSQI' in df.columns:
        shift_scores = df.groupby('Shift_Rotation')['Total_scorePSQI'].mean().sort_values(ascending=False)
        worst_shift = shift_scores.index[0]
        worst_score = shift_scores.iloc[0]
        insights['worst_shift'] = f"The shift type with the poorest sleep quality is **{worst_shift}** (Avg PSQI: {worst_score:.1f})."
        
    # 3. Risk Chronotype
    if 'Chronotype_Shift' in df.columns and 'Actual_sleep_hours' in df.columns:
        chrono_sleep = df.groupby('Chronotype_Shift')['Actual_sleep_hours'].mean().sort_values()
        risk_chrono = chrono_sleep.index[0]
        hours = chrono_sleep.iloc[0]
        insights['risk_chronotype'] = f"**{risk_chrono}** chronotypes get the least sleep on average ({hours:.1f} hours)."

    # 4. Nutrition Insights
    if 'Eating_window_workday' in numeric_df.columns and 'Total_scorePSQI' in numeric_df.columns:
        corr_nut = numeric_df['Eating_window_workday'].corr(numeric_df['Total_scorePSQI'])
        direction = "worse" if corr_nut > 0 else "better" # Positive correlation with PSQI means higher score (worse sleep)
        insights['nutrition'] = f"A larger eating window on workdays is associated with **{direction}** sleep quality (Correlation: {corr_nut:.2f})."

    return insights
