import pandas as pd

# load data
df=pd.read_csv("data/rainfall_data_2015_2025.csv")

# convert string date to real date
df['Date']=pd.to_datetime(df['Date'])
df=df.sort_values('Date')

# lag features
df['lag1']=df['rainfall'].shift(1)
df['lag2']=df['rainfall'].shift(2)
df['lag3']=df['rainfall'].shift(3)

# delete NAN value
df=df.dropna()

# define X,y
X=df[['lag1','lag2','lag3']]
y=df['rainfall']