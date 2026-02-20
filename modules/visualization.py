import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def time_to_decimal(time_val):
    """Converts time string (HH:MM) or datetime to decimal hours."""
    if pd.isna(time_val):
        return None
    try:
        # If it's already a number, assume it's decimal or needs no conversion
        if isinstance(time_val, (int, float)):
            return time_val
        
        # If string, try to parse
        time_str = str(time_val).strip()
        if ':' in time_str:
            h, m = map(int, time_str.split(':')[:2])
            return h + m/60.0
        else:
             # Handle cases like "23" or "2300" if strictly formatted, but assuming HH:MM
             return float(time_str)
    except:
        return None

def plot_demographics(df):
    """Generates charts for Section 2A: Sociodemographic Insights"""
    df = df.copy()  # Prevent SettingWithCopyWarning
    charts = {}
    
    # 1. Bar chart — Sleep Quality by Gender
    # Group by Gender and calculate average Total_sleep_quality or distribution
    # User asked for "Sleep Quality by Gender". Box or Bar? Bar usually implies aggregation.
    # Let's show average Total_scorePSQI (Lower is better) or Total_sleep_quality count?
    # "Total_sleep_quality" is likely categorical or score. 
    # Let's assume categorical distribution for stacked bar, or Avg PSQI.
    # I'll do a box plot or violin if distribution is needed, but user said Bar.
    # I'll do Avg Total_scorePSQI by Gender.
    if 'Total_scorePSQI' in df.columns and 'Gender' in df.columns:
        avg_score = df.groupby('Gender')['Total_scorePSQI'].mean().reset_index()
        fig1 = px.bar(avg_score, x='Gender', y='Total_scorePSQI', title='Average PSQI Score by Gender (Lower is Better)', color='Gender')
        charts['gender_sleep'] = fig1

    # 2. Histogram — Age Distribution
    if 'Age' in df.columns:
        fig2 = px.histogram(df, x='Age', nbins=20, title='Age Distribution of Participants', color_discrete_sequence=['#636EFA'])
        charts['age_dist'] = fig2

    # 3. Box plot — BMI vs Sleep hours
    # Bin BMI into categories if it's continuous? No, Box plot usually x=Category, y=Continuous.
    # If BMI is continuous, maybe Scatter? Or bin it. 
    # "Box plot — BMI vs Sleep hours". Likely BMI Category vs Sleep Hours.
    # Let's create BMI categories if needed, or if BMI is a category column?
    # Usually BMI is numeric. I'll create a bin.
    if 'BMI' in df.columns and 'Actual_sleep_hours' in df.columns:
        df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        fig3 = px.box(df, x='BMI_Category', y='Actual_sleep_hours', title='Sleep Hours Distribution by BMI Category', color='BMI_Category')
        charts['bmi_sleep'] = fig3

    # 4. Stacked bar — Education vs Sleep Quality
    # Sleep Quality (Good/Poor) is derived from PSQI usually (>5 is poor).
    if 'Highest_Education' in df.columns:
        if 'Sleep_Quality_Bin' not in df.columns and 'Total_scorePSQI' in df.columns:
            df['Sleep_Quality_Bin'] = df['Total_scorePSQI'].apply(lambda x: 'Poor (>5)' if x > 5 else 'Good (<=5)')
        
        if 'Sleep_Quality_Bin' in df.columns:
            quality_counts = df.groupby(['Highest_Education', 'Sleep_Quality_Bin']).size().reset_index(name='Count')
            fig4 = px.bar(quality_counts, x='Highest_Education', y='Count', color='Sleep_Quality_Bin', title='Sleep Quality Distribution by Education', barmode='stack')
            charts['edu_sleep'] = fig4

    return charts

def plot_shift_work(df):
    """Generates charts for Section 2B: Shift-Work & Sleep Patterns"""
    df = df.copy()  # Prevent SettingWithCopyWarning
    charts = {}
    
    # 1. Box plot — Actual_sleep_hours by Shift_Rotation
    if 'Actual_sleep_hours' in df.columns and 'Shift_Rotation' in df.columns:
        fig1 = px.box(df, x='Shift_Rotation', y='Actual_sleep_hours', color='Shift_Rotation', title='Sleep Hours by Shift Rotation')
        charts['shift_sleep_hours'] = fig1

    # 2. Heatmap — Median Bedtime vs Chronotype_Shift vs Shift_Rotation
    # x=Chronotype_Shift, y=Shift_Rotation, z=Median Bedtime
    if 'Bedtime' in df.columns and 'Shift_Rotation' in df.columns and 'Chronotype_Shift' in df.columns:
        df['Bedtime_Dec'] = df['Bedtime'].apply(time_to_decimal)
        pivot = df.pivot_table(index='Shift_Rotation', columns='Chronotype_Shift', values='Bedtime_Dec', aggfunc='median')
        fig2 = px.imshow(pivot, title='Median Bedtime (Hour) by Shift & Chronotype', text_auto='.1f', color_continuous_scale='Viridis')
        charts['bedtime_heatmap'] = fig2

    # 3. Scatter — Sleep_latency vs Sleep hours colored by Industry
    if 'Sleep_latency' in df.columns and 'Actual_sleep_hours' in df.columns and 'Industry' in df.columns:
        fig3 = px.scatter(df, x='Actual_sleep_hours', y='Sleep_latency', color='Industry', title='Sleep Latency vs Hours by Industry', hover_data=['Subject_ID'])
        charts['latency_scatter'] = fig3

    # 4. Line chart — Sleep hours trend across different shift types
    # "Trend" usually implies time. But here maybe comparison?
    # Or maybe user means "Trend across shift types" as a line categorical plot?
    # I'll compute Mean Sleep Hours per Shift Rotation and plot as a line.
    if 'Actual_sleep_hours' in df.columns and 'Shift_Rotation' in df.columns:
        trend_data = df.groupby('Shift_Rotation')['Actual_sleep_hours'].mean().reset_index()
        fig4 = px.line(trend_data, x='Shift_Rotation', y='Actual_sleep_hours', markers=True, title='Average Sleep Hours Trend across Shift Types')
        charts['shift_trend'] = fig4

    return charts

def plot_chronotype(df):
    """Generates charts for Section 2C: Chronotype Analysis"""
    df = df.copy()  # Prevent SettingWithCopyWarning
    charts = {}
    
    # 1. Pie chart — Chronotype_Shift distribution
    if 'Chronotype_Shift' in df.columns:
        counts = df['Chronotype_Shift'].value_counts().reset_index()
        counts.columns = ['Chronotype_Shift', 'Count'] # Rename for safety
        fig1 = px.pie(counts, names='Chronotype_Shift', values='Count', title='Distribution of Shift Chronotypes')
        charts['chrono_pie'] = fig1

    # 2. Histogram — ScoreMEQ distribution
    if 'ScoreMEQ' in df.columns:
        fig2 = px.histogram(df, x='ScoreMEQ', nbins=20, title='Distribution of MEQ Scores', color_discrete_sequence=['#00CC96'])
        charts['meq_hist'] = fig2

    # 3. Box plot — Actual_sleep_hours by Chronotype_MEQ
    if 'Actual_sleep_hours' in df.columns and 'Chronotype_MEQ' in df.columns:
        fig3 = px.box(df, x='Chronotype_MEQ', y='Actual_sleep_hours', color='Chronotype_MEQ', title='Sleep Hours by MEQ Chronotype')
        charts['chrono_sleep_box'] = fig3

    # 4. Scatter — MSFSC vs Sleep hours
    # MSFSC = Mid-Sleep on Free Days (Sleep Corrected). Not in main list but maybe available?
    # User said "Main dataset includes... Chronotypes dataset... MSFSC"
    # I will assume it's merged or available.
    if 'MSFSC' in df.columns and 'Actual_sleep_hours' in df.columns:
        # MSFSC is usually time. Convert to decimal.
        df['MSFSC_Dec'] = df['MSFSC'].apply(time_to_decimal)
        # Handle cases where MSFSC > 24 (e.g. 26:00 = 02:00 next day)
        # For plotting linear relationship, 26.0 is fine.
        fig4 = px.scatter(df, x='MSFSC_Dec', y='Actual_sleep_hours', title='Sleep Duration vs Mid-Sleep (MSFSC)', trendline="ols")
        charts['msfsc_scatter'] = fig4

    return charts

def plot_nutrition(df):
    """Generates charts for Section 2D: Chrononutrition Behavior"""
    df = df.copy()  # Prevent SettingWithCopyWarning
    charts = {}
    
    # 1. Bar chart — Largest_mealtime vs Sleep Quality
    # Largest_mealtime is time. We assume user wants to see if late eaters have worse sleep.
    # We can bin mealtime or just sort categorical if it's text.
    # If it's time, lets bin it into hours or periods.
    if 'Largest_mealtime' in df.columns and 'Total_scorePSQI' in df.columns:
        df['Mealtime_Dec'] = df['Largest_mealtime'].apply(time_to_decimal)
        # Create categories: Early (<12), Mid (12-18), Late (18-21), Very Late (>21)
        # Or just plot Median PSQI by Hour of Meal
        if df['Mealtime_Dec'].notnull().any():
            df['Meal_Hour'] = df['Mealtime_Dec'].apply(lambda x: int(x) if pd.notnull(x) else np.nan)
            avg_psqi = df.groupby('Meal_Hour')['Total_scorePSQI'].mean().reset_index()
            fig1 = px.bar(avg_psqi, x='Meal_Hour', y='Total_scorePSQI', title='Avg PSQI Score by Largest Meal Hour', labels={'Meal_Hour': 'Hour of Day'})
            charts['meal_psqi'] = fig1

    # 2. Scatter — Eating_window_workday vs Actual_sleep_hours
    if 'Eating_window_workday' in df.columns and 'Actual_sleep_hours' in df.columns:
        fig2 = px.scatter(df, x='Eating_window_workday', y='Actual_sleep_hours', title='Eating Window (Workday) vs Sleep Hours', trendline="ols")
        charts['window_sleep'] = fig2

    # 3. Scatter — Last_eating_workday vs Bedtime
    if 'Last_eating_workday' in df.columns and 'Bedtime' in df.columns:
        df['Last_Eat_Dec'] = df['Last_eating_workday'].apply(time_to_decimal)
        df['Bedtime_Dec'] = df['Bedtime'].apply(time_to_decimal)
        fig3 = px.scatter(df, x='Last_Eat_Dec', y='Bedtime_Dec', title='Last Eating Time vs Bedtime (Workday)', labels={'Last_Eat_Dec': 'Last Meal Time (Hr)', 'Bedtime_Dec': 'Bedtime (Hr)'})
        charts['eat_bed_scatter'] = fig3

    # 4. Compare Workday vs Freeday eating patterns
    # Box plot comparison of Eating_window or Last_eating?
    # "Eating_window_workday" vs "Eating_window_freeday"
    if 'Eating_window_workday' in df.columns and 'Eating_window_freeday' in df.columns:
        # Melt for comparison
        melted = df.melt(value_vars=['Eating_window_workday', 'Eating_window_freeday'], var_name='Day_Type', value_name='Hours')
        fig4 = px.box(melted, x='Day_Type', y='Hours', color='Day_Type', title='Eating Window Duration: Workday vs Freeday')
        charts['window_compare'] = fig4

    return charts

def plot_confusion_matrix(cm, labels=["Good Sleep", "Poor Sleep"]):
    """Plots a Confusion Matrix as a Heatmap."""
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale="Blues"
    )
    fig.update_layout(title="Confusion Matrix")
    return fig

from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_test, y_prob):
    """Plots ROC Curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC = {roc_auc:.3f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=500, height=400
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    return fig
