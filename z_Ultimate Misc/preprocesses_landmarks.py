import re
import pandas as pd

data = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\aligned_twisthand_data_20230801-165125.csv")

# Extracting the x, y, and z coordinates from the string
def extract_coordinates(landmark_str):
    x_values = re.findall(r'x: ([\d.-]+)', landmark_str)
    y_values = re.findall(r'y: ([\d.-]+)', landmark_str)
    z_values = re.findall(r'z: ([\d.-]+)', landmark_str)
    
    return x_values, y_values, z_values

# Applying the extraction function
data['x_values'], data['y_values'], data['z_values'] = zip(*data['Hand 1 Landmarks'].map(extract_coordinates))

# Expanding the lists into separate columns
for i in range(21):  # Assuming 21 landmarks
    data[f'x_{i}'] = data['x_values'].str[i].astype(float)
    data[f'y_{i}'] = data['y_values'].str[i].astype(float)
    data[f'z_{i}'] = data['z_values'].str[i].astype(float)

# Dropping the original and intermediate columns
data.drop(columns=['Hand 1 Landmarks', 'Hand 2 Landmarks', 'x_values', 'y_values', 'z_values'], inplace=True)

#changes the name of a column
#data.rename(columns={"Unnamed: 0": "Index"}, inplace=True)

data.to_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\twisthand_dataframe.csv", index=False)


