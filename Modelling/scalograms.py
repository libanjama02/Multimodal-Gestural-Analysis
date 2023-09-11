import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd

df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\1stAug\twisthand_imu_data_20230801-165125.csv")
#df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\1stAug\openclose_imu_data_20230801-165524.csv")


df_quaternion = df[['w', 'x', 'y', 'z']]
df_acceleration = df[['ax', 'ay', 'az']]

"""
# Function to generate and plot scalograms using CWT
def plot_scalogram(data, title, col_name):
    scales = np.arange(1, 64) # needs to be tuned
    coefficients, frequencies = pywt.cwt(data, scales, 'cmor')
    plt.imshow(np.abs(coefficients), aspect='auto', extent=[0, len(data), 1, 128], cmap='viridis', interpolation='bilinear')
    plt.colorbar(label='Magnitude')
    plt.title(f'Scalogram of {title} - {col_name}')
    plt.xlabel('Sample Count')
    plt.ylabel('Scale')
    plt.show()

# Generate and plot scalograms for Quaternion components
for col in df_quaternion.columns:
    plot_scalogram(df_quaternion[col], 'Quaternion', col)

# Generate and plot scalograms for Acceleration components
for col in df_acceleration.columns:
    plot_scalogram(df_acceleration[col], 'Acceleration', col)

"""
########

# Modified function to generate and plot scalograms using CWT in subplots
def plot_scalograms_side_by_side(dataframes, titles, col_names):
    fig, axes = plt.subplots(1, len(dataframes), figsize=(20, 4))  # Adjust figsize as needed

    for i, (df, title, col_name) in enumerate(zip(dataframes, titles, col_names)):
        scales = np.arange(1, 64) #needs to be tuned
        coefficients, frequencies = pywt.cwt(df, scales, 'cmor') #change wave
        im = axes[i].imshow(np.abs(coefficients), aspect='auto', extent=[0, len(df), 1, 128], cmap='viridis', interpolation='bilinear') 
        axes[i].set_title(f'{title} Scalogram - {col_name}')
        axes[i].set_xlabel('Sample Count')
        axes[i].set_ylabel('Scale')

    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Magnitude')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the layout so everything fits
    plt.show()

# Generate and plot scalograms for Quaternion and Acceleration components side-by-side
plot_scalograms_side_by_side(
    dataframes=[df_quaternion[col] for col in df_quaternion.columns] + [df_acceleration[col] for col in df_acceleration.columns],
    titles=['Quaternion'] * 4 + ['Acceleration'] * 3,
    col_names=df_quaternion.columns.tolist() + df_acceleration.columns.tolist()
)    