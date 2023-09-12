import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\Gestures\twisthand\!twisthand_dataframe.csv")


#Iterates X, Y, Z coordinates against time for each landmark (close plot to see next one)
landmark_ids = list(range(21))
for landmark_id in landmark_ids:
    plt.figure(figsize=(12, 6))
    plt.plot(df[f'x_{landmark_id}'], label=f'x_{landmark_id}')
    plt.plot(df[f'y_{landmark_id}'], label=f'y_{landmark_id}')
    plt.plot(df[f'z_{landmark_id}'], label=f'z_{landmark_id}')
    plt.title(f'Landmark {landmark_id} Position Over Time/Frame ID')
    plt.xlabel('Frame')
    plt.ylabel('Position')
    plt.legend()
    plt.show()

#Quaternion data against time
plt.figure(figsize=(12, 6))
plt.plot(df['x'], label='x')
plt.plot(df['y'], label='y')
plt.plot(df['z'], label='z')
plt.plot(df['w'], label='w')
plt.title('Quaternion Over Time/Frame ID')
plt.xlabel('Frame')
plt.ylabel('Quaternion Value')
plt.legend()
plt.show()

# Acceleration data against time
plt.figure(figsize=(12, 6))
plt.plot(df['ax'], label='ax')
plt.plot(df['ay'], label='ay')
plt.plot(df['az'], label='az')
plt.title('Acceleration Over Time/Frame ID')
plt.xlabel('Frame')
plt.ylabel('Acceleration')
plt.legend()
plt.show()


#Velocity of X, Y, Z for Landmark 0 (using the difference method)
df['vx'] = df['x_0'].diff().fillna(0)
df['vy'] = df['y_0'].diff().fillna(0)
df['vz'] = df['z_0'].diff().fillna(0)

plt.figure(figsize=(12, 6))
plt.plot(df['vx'], label='vx')
plt.plot(df['vy'], label='vy')
plt.plot(df['vz'], label='vz')
plt.title('Velocity Over Time/Frame ID (For Landmark 0)')
plt.xlabel('Frame')
plt.ylabel('Velocity')
plt.legend()
plt.show()


#Heatmap of landmark positions over time
plt.figure(figsize=(12, 6))
plt.imshow(df[[f'x_{i}' for i in landmark_ids]].T, aspect='auto', cmap='hot')
plt.colorbar(label='Position Value')
plt.title('Landmark X-positions Over Time/Frame ID')
plt.xlabel('Frame')
plt.ylabel('Landmark ID')
plt.show()

plt.figure(figsize=(12, 6))
plt.imshow(df[[f'y_{i}' for i in landmark_ids]].T, aspect='auto', cmap='hot')
plt.colorbar(label='Position Value')
plt.title('Landmark Y-positions Over Time/Frame ID')
plt.xlabel('Frame')
plt.ylabel('Landmark ID')
plt.show()

plt.figure(figsize=(12, 6))
plt.imshow(df[[f'z_{i}' for i in landmark_ids]].T, aspect='auto', cmap='hot')
plt.colorbar(label='Position Value')
plt.title('Landmark Z-positions Over Time/Frame ID')
plt.xlabel('Frame')
plt.ylabel('Landmark ID')
plt.show()


#Heatmap of acceleration data over time
plt.figure(figsize=(12, 6))
plt.imshow(df[['ax', 'ay', 'az']].T, aspect='auto', cmap='hot')
plt.colorbar(label='Acceleration Value')
plt.title('Acceleration Over Time/Frame ID')
plt.xlabel('Frame')
plt.ylabel('Acceleration Axis')
#plt.yticks([0, 1, 2], ['ax', 'ay', 'az'])
plt.show()

