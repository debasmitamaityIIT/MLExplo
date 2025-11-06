import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# === STEP 1: Load the -10°C .mat file ===
mat_data = sio.loadmat("06-08-17_10.21 3740_Charge1a.mat")

# Explore structure
print(mat_data.keys())

# === STEP 2: Extract important arrays ===
# NASA datasets usually store these under:
# 'meas' -> 'Voltage_measured', 'Current_measured', 'Temperature_measured', 'Time'
data = mat_data['meas']

voltage = data['Voltage_measured'][0,0].flatten()
current = data['Current_measured'][0,0].flatten()
temp = data['Temperature_measured'][0,0].flatten()
time = data['Time'][0,0].flatten()

# === STEP 3: Compute Heat Generation (Simplified Q = I * (V - Voc_est)) ===
# If Voc is not directly available, approximate with rolling average voltage
Voc_est = pd.Series(voltage).rolling(window=50, min_periods=1).mean()
Q_gen = current * (voltage - Voc_est)

df = pd.DataFrame({
    'Voltage': voltage,
    'Current': current,
    'Temperature': temp,
    'HeatGen': Q_gen
})
print(df.head())

# === STEP 4: Prepare data for LSTM ===
features = df[['Voltage', 'Current', 'Temperature']].values
labels = df['HeatGen'].values.reshape(-1, 1)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
features_scaled = scaler_x.fit_transform(features)
labels_scaled = scaler_y.fit_transform(labels)

# Create sequences
seq_len = 20
X, y = [], []
for i in range(len(features_scaled) - seq_len):
    X.append(features_scaled[i:i+seq_len])
    y.append(labels_scaled[i+seq_len])
X, y = np.array(X), np.array(y)

# === STEP 5: Define and Train LSTM ===
model = Sequential([
    LSTM(64, input_shape=(seq_len, 3), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=15, batch_size=32, verbose=1)

# === STEP 6: Predict and plot ===
y_pred_scaled = model.predict(X)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = labels[seq_len:]

plt.figure(figsize=(10,5))
plt.plot(y_true, label='True Heat Generation', linewidth=2)
plt.plot(y_pred, label='Predicted Heat Generation', linestyle='dashed')
plt.legend()
plt.xlabel('Time step')
plt.ylabel('Heat Generation (approx)')
plt.title('LSTM Prediction vs True Heat Generation (-10°C)')
plt.show()
