"""
Circadian Rhythm Analysis Module for AI Sleep Lab

This module provides circadian-specific feature engineering and analysis
for understanding how biological rhythms affect sleep quality.

Key features:
- Social Jetlag calculation
- Circadian Misalignment metrics
- Cosinor rhythm analysis
- Phase Delay Index

Designed to support publication on "ML for Understanding Circadian Rhythm"
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy.optimize import curve_fit
from scipy import stats
from dataclasses import dataclass
import warnings


@dataclass
class CosinorResult:
    """Container for cosinor analysis results."""
    mesor: float           # Rhythm-adjusted mean
    amplitude: float       # Peak-to-trough / 2
    acrophase: float       # Phase of peak (hours)
    acrophase_time: str    # Peak time as HH:MM
    r_squared: float       # Goodness of fit
    
    def __str__(self) -> str:
        return (f"Cosinor: Mesor={self.mesor:.2f}, Amplitude={self.amplitude:.2f}, "
                f"Acrophase={self.acrophase_time} (R²={self.r_squared:.3f})")


def time_to_minutes(time_val) -> Optional[float]:
    """
    Convert time value to minutes since midnight.
    
    Handles various input formats:
    - HH:MM string
    - datetime.time object  
    - Integer (assumed to be minutes already)
    - Float (assumed to be hours)
    
    Args:
        time_val: Time value in various formats
    
    Returns:
        Minutes since midnight, or None if parsing fails
    """
    if pd.isna(time_val):
        return None
    
    try:
        # Already numeric
        if isinstance(time_val, (int, float)):
            # Assume hours if < 24, minutes if >= 24
            if time_val < 24:
                return time_val * 60
            return float(time_val)
        
        # datetime.time object
        if hasattr(time_val, 'hour'):
            return time_val.hour * 60 + time_val.minute
        
        # String parsing
        time_str = str(time_val).strip()
        if ':' in time_str:
            parts = time_str.split(':')
            h, m = int(parts[0]), int(parts[1])
            return h * 60 + m
        
        # Try as float (hours)
        val = float(time_str)
        return val * 60 if val < 24 else val
        
    except (ValueError, TypeError):
        return None


def minutes_to_hours(minutes: float) -> float:
    """Convert minutes to decimal hours."""
    return minutes / 60.0


def normalize_time(minutes: float, reference_midnight: bool = True) -> float:
    """
    Normalize time to handle wrap-around midnight issues.
    
    For sleep analysis, times after midnight (e.g., 01:00) should be 
    treated as "late" (25:00 = 01:00).
    
    Args:
        minutes: Time in minutes since midnight
        reference_midnight: If True, times before 6:00 AM are treated as next day
    
    Returns:
        Normalized time in minutes
    """
    if reference_midnight and minutes < 360:  # Before 6:00 AM
        return minutes + 1440  # Add 24 hours
    return minutes


# ============================================================================
# CIRCADIAN FEATURE ENGINEERING
# ============================================================================

def calculate_social_jetlag(midsleep_workday: float, 
                            midsleep_freeday: float) -> float:
    """
    Calculate Social Jetlag (SJL).
    
    Social jetlag is the difference between mid-sleep times on work days
    vs free days, reflecting circadian misalignment due to social obligations.
    
    Formula: SJL = MSFsc - MSW
    Where:
        MSFsc = Mid-Sleep on Free days (sleep corrected)
        MSW = Mid-Sleep on Workdays
    
    Args:
        midsleep_workday: Mid-sleep time on workdays (minutes since midnight)
        midsleep_freeday: Mid-sleep time on free days (minutes since midnight)
    
    Returns:
        Social jetlag in hours (positive = later on free days)
    """
    if pd.isna(midsleep_workday) or pd.isna(midsleep_freeday):
        return np.nan
    
    # Normalize times (handle post-midnight sleep)
    msw = normalize_time(midsleep_workday)
    msf = normalize_time(midsleep_freeday)
    
    # Calculate difference in hours
    sjl_minutes = msf - msw
    return sjl_minutes / 60.0


def calculate_midsleep(bedtime: float, wake_time: float) -> float:
    """
    Calculate mid-sleep point.
    
    Mid-sleep is the midpoint between sleep onset and wake-up,
    often used as a marker of circadian phase.
    
    Args:
        bedtime: Sleep onset in minutes since midnight
        wake_time: Wake-up time in minutes since midnight
    
    Returns:
        Mid-sleep point in minutes
    """
    if pd.isna(bedtime) or pd.isna(wake_time):
        return np.nan
    
    # Normalize for overnight sleep
    bed_norm = normalize_time(bedtime)
    wake_norm = normalize_time(wake_time)
    
    # Ensure wake is after bed
    if wake_norm < bed_norm:
        wake_norm += 1440  # Add 24 hours
    
    midsleep = (bed_norm + wake_norm) / 2
    
    # Convert back to standard time if > 24 hours
    if midsleep >= 1440:
        midsleep -= 1440
    
    return midsleep


def calculate_phase_delay_index(bedtime: float, 
                                reference_hour: float = 22.0) -> float:
    """
    Calculate Phase Delay Index (PDI).
    
    Measures how "delayed" a person's sleep schedule is relative to
    a reference early bedtime (default 10 PM).
    
    Higher PDI = more delayed/evening-shifted schedule.
    
    Args:
        bedtime: Bedtime in minutes since midnight
        reference_hour: Reference "early" bedtime in hours (default 22:00)
    
    Returns:
        Phase delay in hours
    """
    if pd.isna(bedtime):
        return np.nan
    
    # Normalize bedtime
    bed_norm = normalize_time(bedtime)
    reference_minutes = reference_hour * 60
    
    # Calculate delay (positive = later than reference)
    delay_minutes = bed_norm - reference_minutes
    return delay_minutes / 60.0


def calculate_circadian_misalignment(chronotype_meq: float,
                                      bedtime: float,
                                      shift_type: int) -> float:
    """
    Calculate Circadian Misalignment Score.
    
    Measures the degree of mismatch between a person's natural chronotype
    and their actual sleep schedule (influenced by shift work).
    
    Higher score = greater misalignment.
    
    Args:
        chronotype_meq: MEQ-derived chronotype (1-5 scale, 1=evening, 5=morning)
        bedtime: Actual bedtime in minutes
        shift_type: Shift rotation type (1=day, 2=rotating, 3=night)
    
    Returns:
        Misalignment score (0-100 scale)
    """
    if pd.isna(chronotype_meq) or pd.isna(bedtime):
        return np.nan
    
    # Expected optimal bedtime based on chronotype (in hours)
    # Morning types: earlier bedtime (~21:00-22:00)
    # Evening types: later bedtime (~24:00-01:00)
    optimal_bedtime_hours = {
        1: 24.5,   # Definite evening
        2: 23.5,   # Moderate evening
        3: 23.0,   # Intermediate
        4: 22.0,   # Moderate morning
        5: 21.5    # Definite morning
    }
    
    # Get optimal bedtime (round chronotype to nearest integer)
    chrono_key = max(1, min(5, round(chronotype_meq)))
    optimal_minutes = optimal_bedtime_hours.get(chrono_key, 23.0) * 60
    
    # Calculate deviation
    actual_bed = normalize_time(bedtime)
    deviation = abs(actual_bed - optimal_minutes)
    
    # Factor in shift type (rotating/night shifts compound misalignment)
    shift_multiplier = {1: 1.0, 2: 1.3, 3: 1.5}.get(shift_type, 1.0)
    
    # Normalize to 0-100 scale (cap at 4 hours deviation = 100)
    score = min(100, (deviation / 240) * 100) * shift_multiplier
    
    return min(100, score)


def calculate_chronotype_shift_match(chronotype_shift: int, 
                                      shift_rotation: int) -> int:
    """
    Calculate if chronotype matches work shift pattern.
    
    Optimal matching:
    - Morning chronotype (1) + Day shift (1) = Match
    - Evening chronotype (3) + Night shift (3) = Match
    
    Args:
        chronotype_shift: Circadian preference (1=morning, 2=intermediate, 3=evening)
        shift_rotation: Shift type (1=fixed day, 2=rotating, 3=fixed night/irregular)
    
    Returns:
        1 if matched, 0 if mismatched
    """
    if pd.isna(chronotype_shift) or pd.isna(shift_rotation):
        return np.nan
    
    # Matching logic
    if chronotype_shift == 1 and shift_rotation == 1:  # Morning + Day
        return 1
    elif chronotype_shift == 3 and shift_rotation == 3:  # Evening + Night
        return 1
    elif chronotype_shift == 2:  # Intermediate can adapt
        return 1
    else:
        return 0


def engineer_circadian_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all circadian-engineered features to a DataFrame.
    
    This is the main function to call for feature engineering.
    
    Args:
        df: DataFrame with raw sleep/chronotype data
    
    Returns:
        DataFrame with added circadian features
    """
    df = df.copy()
    
    # Convert time columns to minutes if needed
    time_cols = ['Bedtime', 'Wake_up_time', 'Last_eating_workday', 'Last_eating_freeday']
    for col in time_cols:
        if col in df.columns:
            df[f'{col}_min'] = df[col].apply(time_to_minutes)
    
    # 1. Mid-sleep calculations
    if 'Bedtime_min' in df.columns and 'Wake_up_time_min' in df.columns:
        df['Midsleep'] = df.apply(
            lambda row: calculate_midsleep(row['Bedtime_min'], row['Wake_up_time_min']),
            axis=1
        )
    
    # 2. Phase Delay Index
    if 'Bedtime_min' in df.columns:
        df['Phase_Delay_Index'] = df['Bedtime_min'].apply(calculate_phase_delay_index)
    
    # 3. Circadian Misalignment
    if all(col in df.columns for col in ['Chronotype_MEQ', 'Bedtime_min', 'Shift_Rotation']):
        df['Circadian_Misalignment'] = df.apply(
            lambda row: calculate_circadian_misalignment(
                row['Chronotype_MEQ'], 
                row['Bedtime_min'],
                row['Shift_Rotation']
            ),
            axis=1
        )
    
    # 4. Chronotype-Shift Match
    if 'Chronotype_Shift' in df.columns and 'Shift_Rotation' in df.columns:
        df['Chronotype_Shift_Match'] = df.apply(
            lambda row: calculate_chronotype_shift_match(
                row['Chronotype_Shift'],
                row['Shift_Rotation']
            ),
            axis=1
        )
    
    # 5. Sleep Efficiency Proxy (actual/target sleep)
    if 'Actual_sleep_hours' in df.columns:
        df['Sleep_Efficiency_Proxy'] = df['Actual_sleep_hours'] / 8.0
    
    # 6. Late Eating Flag (eating within 2 hours of bed)
    if 'Last_eating_workday_min' in df.columns and 'Bedtime_min' in df.columns:
        df['Late_Eating_Flag'] = df.apply(
            lambda row: 1 if (
                pd.notna(row['Bedtime_min']) and 
                pd.notna(row['Last_eating_workday_min']) and
                (normalize_time(row['Bedtime_min']) - row['Last_eating_workday_min']) < 120
            ) else 0,
            axis=1
        )
    
    return df


# ============================================================================
# COSINOR ANALYSIS
# ============================================================================

def cosinor_model(t: np.ndarray, mesor: float, amplitude: float, 
                  acrophase: float, period: float = 24.0) -> np.ndarray:
    """
    Cosinor model function for fitting circadian rhythm.
    
    y(t) = M + A * cos(2π * (t - φ) / T)
    
    Where:
        M = Mesor (rhythm-adjusted mean)
        A = Amplitude (peak-to-trough / 2)
        φ = Acrophase (time of peak)
        T = Period (24 hours for circadian)
    
    Args:
        t: Time points (hours)
        mesor: Rhythm-adjusted mean
        amplitude: Half peak-to-trough difference
        acrophase: Phase of peak (hours)
        period: Rhythm period (default 24h for circadian)
    
    Returns:
        Fitted values
    """
    return mesor + amplitude * np.cos(2 * np.pi * (t - acrophase) / period)


def fit_cosinor(times: np.ndarray, values: np.ndarray, 
                period: float = 24.0) -> Optional[CosinorResult]:
    """
    Fit single cosinor model to circadian data.
    
    Args:
        times: Time points in hours (0-24)
        values: Observed values at each time point
        period: Expected rhythm period (default 24h)
    
    Returns:
        CosinorResult object, or None if fitting fails
    """
    # Remove NaN values
    mask = ~(np.isnan(times) | np.isnan(values))
    times_clean = times[mask]
    values_clean = values[mask]
    
    if len(times_clean) < 3:
        return None
    
    # Initial parameter guesses
    mesor_init = np.mean(values_clean)
    amplitude_init = (np.max(values_clean) - np.min(values_clean)) / 2
    acrophase_init = times_clean[np.argmax(values_clean)]
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                lambda t, m, a, p: cosinor_model(t, m, a, p, period),
                times_clean, values_clean,
                p0=[mesor_init, amplitude_init, acrophase_init],
                bounds=(
                    [0, 0, 0],  # Lower bounds
                    [np.inf, np.inf, 24]  # Upper bounds
                ),
                maxfev=5000
            )
            
            mesor, amplitude, acrophase = popt
            
            # Calculate R-squared
            y_pred = cosinor_model(times_clean, mesor, amplitude, acrophase, period)
            ss_res = np.sum((values_clean - y_pred) ** 2)
            ss_tot = np.sum((values_clean - np.mean(values_clean)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Convert acrophase to time string
            acro_hours = int(acrophase)
            acro_mins = int((acrophase - acro_hours) * 60)
            acrophase_time = f"{acro_hours:02d}:{acro_mins:02d}"
            
            return CosinorResult(
                mesor=mesor,
                amplitude=amplitude,
                acrophase=acrophase,
                acrophase_time=acrophase_time,
                r_squared=r_squared
            )
            
    except (RuntimeError, ValueError) as e:
        print(f"Cosinor fitting failed: {e}")
        return None


def analyze_group_rhythms(df: pd.DataFrame, 
                          group_col: str,
                          time_col: str = 'Bedtime_min',
                          value_col: str = 'Actual_sleep_hours') -> Dict[str, CosinorResult]:
    """
    Perform cosinor analysis for different groups.
    
    Useful for comparing circadian rhythm parameters across
    chronotypes, shift types, etc.
    
    Args:
        df: DataFrame with data
        group_col: Column to group by (e.g., 'Chronotype_Shift')
        time_col: Column with time data (minutes)
        value_col: Column with values to analyze
    
    Returns:
        Dictionary of group -> CosinorResult
    """
    results = {}
    
    if time_col not in df.columns or value_col not in df.columns:
        return results
    
    for group_val in df[group_col].dropna().unique():
        group_df = df[df[group_col] == group_val]
        
        times = group_df[time_col].values / 60  # Convert to hours
        values = group_df[value_col].values
        
        result = fit_cosinor(times, values)
        if result:
            results[group_val] = result
    
    return results


# ============================================================================
# SUMMARY STATISTICS FOR PUBLICATION
# ============================================================================

def circadian_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for circadian features.
    
    Designed for publication tables (mean, SD, range by group).
    
    Args:
        df: DataFrame with circadian features
    
    Returns:
        Summary statistics DataFrame
    """
    # Features to summarize
    circadian_cols = [
        'Actual_sleep_hours', 'Sleep_latency', 'Chronotype_MEQ', 'ScoreMEQ',
        'Phase_Delay_Index', 'Circadian_Misalignment', 'Midsleep'
    ]
    
    available_cols = [c for c in circadian_cols if c in df.columns]
    
    if not available_cols:
        return pd.DataFrame()
    
    stats_list = []
    for col in available_cols:
        data = df[col].dropna()
        stats_list.append({
            'Feature': col,
            'N': len(data),
            'Mean': data.mean(),
            'SD': data.std(),
            'Median': data.median(),
            'Min': data.min(),
            'Max': data.max()
        })
    
    return pd.DataFrame(stats_list)


if __name__ == "__main__":
    # Demo
    print("Circadian Analysis Module - Demo")
    print("=" * 50)
    
    # Test social jetlag calculation
    msw = 3 * 60  # 3:00 AM mid-sleep workday
    msf = 5 * 60  # 5:00 AM mid-sleep freeday  
    sjl = calculate_social_jetlag(msw, msf)
    print(f"\nSocial Jetlag: {sjl:.1f} hours")
    
    # Test phase delay
    bedtime = 23 * 60 + 30  # 23:30
    pdi = calculate_phase_delay_index(bedtime)
    print(f"Phase Delay Index: {pdi:.1f} hours from 22:00 reference")
    
    # Test circadian misalignment
    cm = calculate_circadian_misalignment(
        chronotype_meq=4.5,  # Morning type
        bedtime=24 * 60 + 30,  # 00:30 (late for morning type)
        shift_type=2  # Rotating
    )
    print(f"Circadian Misalignment Score: {cm:.1f}/100")
