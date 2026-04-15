# arcgis
import pandas as pd

data={
    'Station': ['Rajshahi','Pabna','Natore','Naogaon','Chapainawabganj'],
    'Latitude': [24.3745,24.0064,24.4206,24.7936,24.5965],
    'Longitude': [88.6042,89.2372,89.0003,88.9318,88.2775],
    'Actual': [1.60,1.42,1.55,1.20,1.75],
    'LSTM': [1.48,1.30,1.50,1.10,1.60],
    'XGBoost': [1.32,1.10,1.20,0.95,1.40]
}

df=pd.DataFrame(data)
df.to_csv("rainfall_gis.csv", index=False)
