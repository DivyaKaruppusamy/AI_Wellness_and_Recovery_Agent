def recovery_score(row):

    score = 100

    if row["sleep_duration_minutes"] < row["sleep_avg_7"] - 60:
        score -= 15

    if row["resting_heart_rate"] > row["rhr_avg_14"] * 1.10:
        score -= 15

    if row["hrv"] < row["hrv_avg_14"] * 0.85:
        score -= 10

    if row["stress_score"] > 70:
        score -= 10

    if row["training_load"] > 80:
        score -= 10

    if score >= 80:
        level = "High"

    elif score >= 60:
        level = "Medium"

    else:
        level = "Low"

    return score, level