import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\!twisthand_dataframe.csv")

#Time Series Plot for chosen landmark (e.g., wrist which is landmark 0)
landmark_index = 0
fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

ax[0].plot(data['Timestamp'], data[f'x_{landmark_index}'], label='X-coordinate', color='r')
ax[0].set_title(f'Landmark {landmark_index} X-coordinate over Time')
ax[0].set_ylabel('X-coordinate')

ax[1].plot(data['Timestamp'], data[f'y_{landmark_index}'], label='Y-coordinate', color='g')
ax[1].set_title(f'Landmark {landmark_index} Y-coordinate over Time')
ax[1].set_ylabel('Y-coordinate')

ax[2].plot(data['Timestamp'], data[f'z_{landmark_index}'], label='Z-coordinate', color='b')
ax[2].set_title(f'Landmark {landmark_index} Z-coordinate over Time')
ax[2].set_ylabel('Z-coordinate')
ax[2].set_xlabel('Time')

plt.tight_layout()
plt.show()

#Time Series Plot for Quaternion data
fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

for i, column in enumerate(['x', 'y', 'z', 'w']):
    ax[i].plot(data['Timestamp'], data[column], label=column)
    ax[i].set_title(f'{column} over Time')
    ax[i].set_ylabel(column)

ax[3].set_xlabel('Time')

plt.tight_layout()
plt.show()

#Time Series Plot for Acceleration data
fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for i, column in enumerate(['ax', 'ay', 'az']):
    ax[i].plot(data['Timestamp'], data[column], label=column)
    ax[i].set_title(f'{column} over Time')
    ax[i].set_ylabel(column)

ax[2].set_xlabel('Time')

plt.tight_layout()
plt.show()
