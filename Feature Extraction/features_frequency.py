import pandas as pd
import numpy as np
#import scipy.signal

def compute_frequency_features(input_path, output_path):
    data = pd.read_csv(input_path)
    
    # Defining the columns to compute FFT and PSD
    columns = [col for col in data.columns if col not in ["Timestamp", "Index", "Frame ID"]]
    
    # Makes dataframe to store features
    freq_features = pd.DataFrame()
    
    for col in columns:
        # Computing FFT and PSD
        fft_values = np.fft.fft(data[col])
        freqs = np.fft.fftfreq(len(fft_values))
        psd_values = np.abs(fft_values) ** 2
        
        # Computing dominant frequency
        dominant_frequency = freqs[np.argmax(psd_values)]
        
        
        # Low (0.1-1 Hz), Mid (1-3 Hz), High (3-10 Hz)
        low_energy = np.sum(psd_values[(freqs >= 0.1) & (freqs <= 1)])
        mid_energy = np.sum(psd_values[(freqs > 1) & (freqs <= 3)])
        high_energy = np.sum(psd_values[(freqs > 3) & (freqs <= 10)])
        
        # Extracting the features
        freq_features['fr_' + col + '_dominant_frequency'] = [dominant_frequency]
        freq_features['fr_' + col + '_low_energy'] = [low_energy]
        freq_features['fr_' + col + '_mid_energy'] = [mid_energy]
        freq_features['fr_' + col + '_high_energy'] = [high_energy]
        
        
        # Could have extracted PSD and FFT but decided not to due to overwhelming amount of data associtaed with it
    
    # Saving results 
    freq_features.to_csv(output_path, index=False)

# Add your file paths here
compute_frequency_features(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Segmentations\twisthand_intermission2.csv", 
                           r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Frequency Features\twisthand_frequency_intermission2.csv")
