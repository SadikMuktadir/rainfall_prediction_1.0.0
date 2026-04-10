# LSTM MODEL

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM,Dense # type: ignore
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

# scale data
scale=MinMaxScaler()
scaleer_data=scale.fit_transform(df[['rainfall']])

# convert 3D
def create_sequences(data,n_steps=3):
    X,y=[],[]
    for i in range(len(data)-n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X),np.array(y)
X_lstm,y_lstm = create_sequences(scaleer_data,3)

# train & test data
train_size = int(len(X_lstm)*0.8)

X_train_lstm = X_lstm[:train_size]
X_test_lstm = X_lstm[train_size:]

y_train_lstm = y_lstm[:train_size]
y_test_lstm = y_lstm[train_size:]

# build model
model_lstm=Sequential()
model_lstm.add(LSTM(50,activation='relu',input_shape=(3,1)))
model_lstm.add(Dense(1))

model_lstm.compile(optimizer='adam', loss='mse')

# train model
model_lstm.fit(X_train_lstm,y_train_lstm, 
               epochs=50,
               batch_size=16,
               verbose=1)

# predict lstm
pred_lstm=model_lstm.predict(X_test_lstm)

# Inverse Scaling
pred_lstm=scale.inverse_transform(pred_lstm)
y_test_lstm=scale.inverse_transform(y_test_lstm)

# Evaluate LSTM
rmse=np.sqrt(mean_squared_error(y_test_lstm,pred_lstm))
mae=mean_absolute_error(y_test_lstm,pred_lstm)
r2=r2_score(y_test_lstm,pred_lstm)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)

# Plot LSTM vs Actual
plt.figure(figsize=(10,5))
plt.plot(y_test_lstm, label="Actual")
plt.plot(pred_lstm, label="LSTM")
plt.legend()
plt.show()