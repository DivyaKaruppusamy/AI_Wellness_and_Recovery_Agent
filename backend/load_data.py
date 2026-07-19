import pandas as pd

df = pd.read_csv("D:\\AI Course\\Capstone_Project\\AI_Wellness_and_Recovery_Agent\\data\\raw\\wearable_metrics.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")

print(df.head())