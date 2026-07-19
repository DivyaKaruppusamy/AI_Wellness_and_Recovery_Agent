import pandas as pd

def calculate_baseline(df):

    df["sleep_avg_7"] = (
        df["sleep_duration_minutes"]
        .rolling(7, min_periods=1)
        .mean()
    )

    df["rhr_avg_14"] = (
        df["resting_heart_rate"]
        .rolling(14, min_periods=1)
        .mean()
    )

    df["hrv_avg_14"] = (
        df["hrv"]
        .rolling(14, min_periods=1)
        .mean()
    )

    return df