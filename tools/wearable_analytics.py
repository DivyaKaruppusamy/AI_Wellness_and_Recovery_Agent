import pandas as pd

def calculate_recovery_metrics(csv_path: str):
    # Load and format the structured dataset
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Isolate today's metrics vs previous historical data
    today_row = df.iloc[-1]
    historical_df = df.iloc[:-1]
    
    # Calculate 7-day or 14-day rolling historical baselines
    base_hr = historical_df['resting_heart_rate'].mean()
    base_hrv = historical_df['hrv'].mean()
    base_sleep = historical_df['sleep_duration_minutes'].mean()
    
    # Execute the explicit project deduction formula
    score = 100
    reasons = []
    
    # 1. Check Sleep Drop
    if today_row['sleep_duration_minutes'] < (base_sleep - 60):
        score -= 15
        reasons.append("Sleep duration is significantly below baseline")
    elif today_row['sleep_duration_minutes'] < (base_sleep - 30):
        score -= 10
        reasons.append("Sleep duration is slightly below baseline")
        
    # 2. Check Elevated Resting Heart Rate
    if today_row['resting_heart_rate'] > (base_hr * 1.10):
        score -= 15
        reasons.append("Resting heart rate is over 10% higher than average")
        
    # 3. Check Depressed HRV
    if today_row['hrv'] < (base_hrv * 0.85):
        score -= 10
        reasons.append("Heart Rate Variability (HRV) is abnormally depressed")
        
    # 4. Check Spiked Stress Metrics
    if today_row['stress_score'] > 60:
        score -= 10
        reasons.append("Daily psychological stress track is elevated")
        
    # Classify Readiness Tier
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
        "today_metrics": today_row.to_dict(),
        "baselines": {"avg_hr": base_hr, "avg_hrv": base_hrv, "avg_sleep": base_sleep}
    }
