import pandas as pd
import streamlit as st
import os

REQUIRED_COLUMNS = [
    "Subject_ID", "Gender", "Age", "Height", "Weight", "BMI", "Ethnicity",
    "Marital_Status", "Household_Income", "Highest_Education", "Industry",
    "Shift_Rotation", "Disease", "Smoking/Vaping", "Part-time",
    "Chronotype_Shift", "Chronotype_MEQ", "ScoreMEQ", "Bedtime",
    "Sleep_latency", "Actual_sleep_hours", "Wake_up_time",
    "PSQI1", "PSQI2", "PSQI3", "PSQI4", "PSQI5", "PSQI6", "PSQI7",
    "Total_scorePSQI", "Total_sleep_quality", "Breakfast_skipping",
    "Largest_mealtime", "Last_eating_workday", "Last_eating_freeday",
    "Eating_window_workday", "Eating_window_freeday"
]

@st.cache_data
def load_data(file_path=None):
    """
    Loads data from CSV or Excel. 
    If file_path is None, looks for default paths.
    """
    df = None
    
    # potential paths (Prioritizing user's specified file)
    paths_to_check = [
        "data/DATA_COMPILATION.xlsx",
        "data/DATA COMPILATION.xlsx",
        "DATA COMPILATION.xlsx",
        "DATA_COMPILATION.xlsx",
        "/Users/kowshik/Documents/Main Project/DATA COMPILATION.xlsx",
        "data/sleep_data.csv"
    ]
    
    # Dynamic search for any xlsx in current dir
    for file in os.listdir('.'):
        if file.endswith(".xlsx") and not file.startswith("~$"):
            paths_to_check.insert(0, file)
            
    # Dynamic search in data/ dir
    if os.path.isdir('data'):
        for file in os.listdir('data'):
            if file.endswith(".xlsx") and not file.startswith("~$"):
                paths_to_check.insert(0, os.path.join('data', file))

    if file_path:
        paths_to_check.insert(0, file_path)

    for path in paths_to_check:
        if os.path.exists(path):
            try:
                if path.endswith(".csv"):
                    df = pd.read_csv(path)
                else:
                    # Robust Excel Loading: Check for multiple sheets
                    xls = pd.ExcelFile(path, engine='openpyxl')
                    sheet_names = xls.sheet_names
                    st.sidebar.info(f"Sheets found: {sheet_names}")
                    
                    if len(sheet_names) > 1:
                        # Attempt to merge all sheets on Subject_ID
                        df = pd.read_excel(path, sheet_name=sheet_names[0], engine='openpyxl')
                        if "Subject_ID" in df.columns:
                            df['Subject_ID'] = df['Subject_ID'].astype(str).str.strip()
                            
                        for sheet in sheet_names[1:]:
                            try:
                                next_df = pd.read_excel(path, sheet_name=sheet, engine='openpyxl')
                                if "Subject_ID" in next_df.columns:
                                    next_df['Subject_ID'] = next_df['Subject_ID'].astype(str).str.strip()
                                    # Merge, handling suffixes if columns overlap
                                    df = pd.merge(df, next_df, on="Subject_ID", how="outer", suffixes=('', '_dup'))
                                    # Drop duplicate columns if any created
                                    df = df.loc[:, ~df.columns.str.endswith('_dup')]
                            except Exception as e:
                                st.warning(f"Could not merge sheet '{sheet}': {e}")
                    else:
                        df = pd.read_excel(path, engine='openpyxl')
                
                # CLEANUP: Strip whitespace from column names
                if df is not None:
                    df.columns = df.columns.str.strip()
                    
                st.sidebar.success(f"Loaded: {path}")
                st.sidebar.write(f"Columns: {len(df.columns)}") # Debug info
                break
            except Exception as e:
                st.error(f"Error loading {path}: {e}")
                continue
    
    if df is None:
        st.error("Dataset not found.")
        st.write(f"Current Working Directory: {os.getcwd()}")
        st.write(f"Files found: {os.listdir('.')}")
        return None

    # Column Validation
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in dataset: {missing_cols}")
        st.write("Available columns:", df.columns.tolist())
        return None

    return df

def preprocess_data(df):
    """
    Basic preprocessing for visualization and modeling.
    """
    df = df.copy()
    
    # Ensure numeric columns are numeric
    numeric_cols = [
        "Age", "Height", "Weight", "BMI", "ScoreMEQ", "Sleep_latency",
        "Actual_sleep_hours", "Total_scorePSQI", "Eating_window_workday",
        "Eating_window_freeday"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Convert problematic columns to string to avoid Arrow confusion
    # MSFSC often contains mixed datetime/time objects and strings
    problem_cols = ['MSFSC', 'Bedtime', 'Wake_up_time', 'Last_eating_workday', 'Last_eating_freeday', 'Disease']
    for col in problem_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df
