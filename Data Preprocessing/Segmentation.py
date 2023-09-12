import pandas as pd

def segment_data_by_index(input_path, start_index, end_index, output_path):
    """
    Manual Segmentation performed comparing timestamps from video ground truth and comparing to Index/Frame ID column from aligned dataframe
    
    Parameters:
    - input_path: Path to the main dataset.
    - start_index: Index to start the segmentation.
    - end_index: Index to end the segmentation.
    - output_path: Path to save the segmented data.
    """
    #reading the file
    data = pd.read_csv(input_path)
    
    #segmenting the data based on the provided indices
    segmented_data = data.iloc[start_index:end_index+1].copy()
    
    #reseting the index for the segmented dataframe output
    segmented_data.reset_index(drop=True, inplace=True)
    
    #adding the 'Index' column
    segmented_data['Index'] = segmented_data.index
    
    #saving the segmented data to the specified path
    segmented_data.to_csv(output_path, index=False)

#using Index column for easier segmentation
segment_data_by_index(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\!twisthand_dataframe.csv", 617, 623,
                       r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Segmentations\twisthand_intermission2.csv")
