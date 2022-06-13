import pandas as pd

df = pd.read_csv("data/mu.csv")

assets = list(df.columns)[1:]
dates = list(df.Date)