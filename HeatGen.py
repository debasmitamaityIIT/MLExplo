import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import glob
import os


base_path = "./battery_data/Panasonic 18650PF Data/-10degC"   # adjust if needed
mat_files = glob.glob(os.path.join(base_path, "*.mat"))
print("Files found:", len(mat_files))


voltage, current, temp = [], [], []

for f in mat_files:
    data = loadmat(f)
    # typical NASA Panasonic .mat structure: data['data'][0,0]
    try:
        d = data['data'][0,0]
        voltage.extend(d['Voltage_measured'][0])
        current.extend(d['Current_measured'][0])
        temp.extend(d['Temperature_measured'][0])
    except KeyError:
        print(f"⚠️ Could not read {f}")

df = pd.DataFrame({
    'Voltage_measured': voltage,
    'Current_measured': current,
    'Temperature_measured': temp
})
print(df.head())


df['Heat'] = df['Current_measured'] * (df['Voltage_measured'] - df['Voltage_measured'].shift(1))
df.dropna(inplace=True)


features = ['Voltage_measured', 'Current_measured', 'Temperature_measured']
X = df[features].values
y = df['Heat'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

def make_seq(X, y, step=15):
    Xs, ys = [], []
    for i in range(len(X)-step):
        Xs.append(X[i:i+step])
        ys.append(y[i+step])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = make_seq(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)


model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)


y_pred = model.predict(X_test)

plt.figure(figsize=(10,5))
plt.plot(y_test[:300], label='Actual Heat')
plt.plot(y_pred[:300], label='Predicted Heat')
plt.xlabel('Time step')
plt.ylabel('Heat generation (Q)')
plt.title('LSTM Heat Generation Prediction (-10°C data)')
plt.legend()
plt.show()
