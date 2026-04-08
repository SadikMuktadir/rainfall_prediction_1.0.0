# XGBOOST MODEL

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

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

# train-test data
train_size=int(len(df)*.8)

X_train=X[:train_size]
X_test=X[train_size:]

y_train=y[:train_size:]
y_test=y[train_size:]

# train XGBoost
model = XGBRegressor()
model.fit(X_train,y_train)

# predict
prediction= model.predict(X_test)

# evaluate model
rmse=np.sqrt(mean_squared_error(y_test,prediction))
mae=mean_absolute_error(y_test,prediction)
r2=r2_score(y_test,prediction)

# plot result
plt.figure(figsize=(10,5))
plt.plot(y_test.values,label="Actual")
plt.plot(prediction,label="Prediction")
plt.legend()
plt.show()

# predict future
last_values = df[['lag1','lag2','lag3']].iloc[-1].values.reshape(1,-1)
future = model.predict(last_values)

# print value
print("Next Rainfall:", future)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)