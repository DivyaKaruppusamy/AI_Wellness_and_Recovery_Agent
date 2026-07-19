import pandas as pd

def calculate_recovery_metrics(csv_path: str, target_date: str = "2026-02-08"):
    df = pd.read_csv(csv_path)
    
    # Strip any trailing whitespace from headers
    df.columns = [c.strip() for c in df.columns]
    
    # Map any shorthand columns from the extracted format safely
    column_mapping = {
        'resting_he': 'resting_heart_rate',
        'sleep_dura': 'sleep_duration_minutes',
        'stress_scoi': 'stress_score'
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Fallback default values if specific columns got clipped during translation
    if 'stress_score' not in df.columns:
        df['stress_score'] = 35  # Healthy default
        
    df['date'] = pd.to_datetime(df['date'].str.strip())
    df = df.sort_values('date')
    
    # Target Row Isolation
    target_dt = pd.to_datetime(target_date)
    today_rows = df[df['date'] == target_dt]
    
    if today_rows.empty:
        today_row = df.iloc[-1]
        historical_df = df.iloc[:-1]
    else:
        today_row = today_rows.iloc[0]
        historical_df = df[df['date'] < target_dt]
        
    if historical_df.empty:
        historical_df = df
        
    # Baseline calculations
    base_hr = historical_df['resting_heart_rate'].mean() if 'resting_heart_rate' in historical_df.columns else historical_df.iloc[:, 3].mean()
    base_hrv = historical_df['hrv'].mean() if 'hrv' in historical_df.columns else historical_df.iloc[:, 6].mean()
    base_sleep = historical_df['sleep_duration_minutes'].mean() if 'sleep_duration_minutes' in historical_df.columns else historical_df.iloc[:, 7].mean()
    
    # Fetch accurate row items safely by position fallback if needed
    t_hr = today_row['resting_heart_rate'] if 'resting_heart_rate' in df.columns else today_row.iloc[3]
    t_hrv = today_row['hrv'] if 'hrv' in df.columns else today_row.iloc[6]
    t_sleep = today_row['sleep_duration_minutes'] if 'sleep_duration_minutes' in df.columns else today_row.iloc[7]
    t_stress = today_row['stress_score'] if 'stress_score' in df.columns else today_row.iloc[10]

    score = 100
    reasons = []
    
    if t_sleep < (base_sleep - 60):
        score -= 15
        reasons.append("Sleep duration is below baseline")
    elif t_sleep < (base_sleep - 30):
        score -= 10
        reasons.append("Sleep duration is below baseline")
        
    if t_hr > (base_hr * 1.10):
        score -= 15
        reasons.append("Resting heart rate is above baseline")
        
    if t_hrv < (base_hrv * 0.85):
        score -= 10
        reasons.append("HRV below baseline")
        
    if t_stress > 60:
        score -= 10
        reasons.append("High stress score")
        
    # Standardised Interface Routing Outputs
    if score >= 80:
        level = "High Recovery"
        routing = "Intense/Normal Workout Allowed"
    elif score >= 60:
        level = "Medium Recovery"
        routing = "Modified Volume Workout Recommended"
    else:
        level = "Low Recovery"
        routing = "Active Recovery/Rest Required"
        
    return {
        "score": max(0, score),
        "level": level,
        "routing": routing,
        "reasons": reasons,
        "today_metrics": {"date": str(target_date), "score": score}
    }
