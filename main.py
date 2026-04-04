import pandas as pd

rainfall_data = pd.read_csv("data/rainfall_data_2015_2025.csv")

rainfall_data["rainfall_lag_1"]=rainfall_data["rainfall"].shift(1)
rainfall_data["rainfall_lag_2"]=rainfall_data["rainfall"].shift(2)
rainfall_data["rainfall_lag_3"]=rainfall_data["rainfall"].shift(3)


train_rainfall_data = rainfall_data[:"2018"]
test_rainfall_data = rainfall_data["2019":]

