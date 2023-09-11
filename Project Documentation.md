

- - - 
## Summary

#### Aim

The aim of this internship project was to explore and prototype methods for classifying and clustering hand gestures in a surgical context. The objectives were aligned with the broader goals of the Surgical Data Science group at KCL, who are working on applying AI and data science to Transsphenoidal Pituitary Endoscopic Surgery.

#### Data Integration 

The first objective was to implement a pipeline that could successfully merge two multimodal data streams. Specifically, the focus was on hand pose landmarks obtained through MediaPipe's framework and Inertial Measurement Unit (IMU) data comprising quaternions and acceleration data collected using an Mbientlab MetaMotionS[Insert Lit Example - https://mbientlab.com/metamotions/], A wearable device that offers real-time and continuous monitoring of motion and environmental sensor data. A time-alignment script was designed to sync these two diverse data streams.

#### Visualization

The second objective involved developing mechanisms for visualizing the recorded gestures. Various visualization scripts were created to understand the quaternion rotations, hand pose landmarks, and their temporal changes. This was not only crucial for data scrutiny but also assisted in feature selection.

#### Machine Learning Prototyping

The final and most significant objective involved prototyping machine learning techniques for the clustering and classification of gestures. Features encapsulating various spatial, temporal, and statistical characteristics were extracted to serve as input to machine learning models. These features were then evaluated using methods commonly used in data science and machine learning in order to assess their effectiveness in capturing the essence of each gesture. Preliminary results showed promising clustering capabilities and classification accuracies ranging from 88.9% to 100%, although it's important to note that the dataset size severely limits the generalizability of these findings. 


This work aims to provide an initial foundation upon which the Surgical Data Science team can build from, offering some avenues for further research in Multimodal Gestural Analysis, such as in integrating other modalities and developing more advanced models for larger applications in the field.

- - - 
## Literature Review

there is a lack of quaternions in gesture recognition, reasons being likely not good sensors. idk confirm with GPT. but here we got a pretty good quaternion and acceleration device recorder? 
most modern literature has an intensive focus on deep learning ngl. ML has fallen out of relevancy, but it's important to build from the principles inih.

shiz like paper 5 shows the emphasis of improving one modality like RGB-D images and having robust gesture recognition. We can take the findings 

The field of gesture recognition has seen significant advancements in recent years across various applications, from healthcare to human-computer interaction. Gesture recognition has proven to be crucial technology in healthcare settings. In particular, there has been a focus in utilizing this technology to aid people with disabilities. One study by X focuses on the development of a non-invasive end to end continuous dynamic gesture recognition system using IMU signal data and surface electromyography (sEMG) for Chinese sign language[1]. Another paper discusses the importance of gesture recognition in enabling real-time interpretation systems for sign language, thus providing an invaluable communication tool for deaf and mute people[3]. The focus may be attributed to ....  (i don't really know what to say here)
The current status of gesture recognition in healthcare leans primarily towards unimodality, dominated by video data particularly. It is pertinent

There has been a shift towards the use of multimodal data, especially sensor fusion. It has emerged as a pivotal technique to enhance gesture recognition systems within certain avenues. For instance, a study proposed a multi-model fall detection system incorporating both vision and sensor data [2] utilizing Random Forest and Long-Term Recurrent Convolutional Networks (LRCN). Another work introduced various innovative sensor fusion approaches like NaiveFuse, KineFuse, and AdaDeepFuse for efficient 3D human pose estimation [9]. These insights are particularly relevant to our work, where fusion of hand pose landmarks and IMU data plays a critical role. 

It is important to highlight the use of quaternion data in gesture recognition. A paper focused on predicting 3D human pose sequences utilized quaternion representations for rotations, effectively addressing issues of error accumulation and discontinuities [10]. This method of using quaternions aligns close with the approach explored within the project, despite there being a low amount of literature focussed on using quaternion data for gesture recognition 


The attempt to integrate hand pose with quaternion and acceleration data for gesture recognition is not a combination frequently explored in existing literature. Especially so given applications to healthcare and surgery in particular. Modelling and curating multimodal datasets is a complex and iterative process. In fields such as surgery, where there is a huge amount of variability and complexity in the operation, where even the most subtlest of gestures have a need to be understood and classified, where noise and data capturing quality is of massive importance, it makes even more sense for why the literature has struggled to lean into this sphere of avenue. 


Whilst most of the recent studies in gesture recognition employ deep learning algorithms and more advanced model architecture, this project starts with foundational machine learning techniques with the goal of understanding and engineering features from the ground up before evolving into more complex models.  

However, the field still presents several challenges, such as the need for efficient real-time analysis and robustness against varied conditions. Our project aims to address these issues by employing a multimodal approach that combines hand pose and quaternion data. This unique combination is relatively underexplored in the literature, which predominantly focuses on leveraging advanced machine learning models and utilizes minimal modalities. 

- - - 
## Experimental Outline

#### Multimodal_Data_Recording.py

The foundation for the experimental process was built upon this Python script. This script served three purposes: 1) capturing hand pose landmarks using MediaPipe's framework. 2) recording IMU data through an MetaMotionS sensor. 3) to run both data collection processes concurrently. In order to circumvent hardware connectivity issues as well as simplify the focus of the project, the experiment and script were deliberately configured to operate with just one sensor and target only one hand.

#### Configuration

The hand pose data was captured using the standard settings for MediaPipe framework. The results provided a robust detection mechanism, however it may be worth tuning the `min_detection_confidence` and `min_tracking_confidence` which I had kept at `0.5`. For the IMU data, a MetaWear sensor with the MAC address `D3:99:34:4D:01:CC` was used. The settings for the sensor were configured to match those used in a previous research study, where real surgical data was collected during Transsphenoidal surgeries on phantom models. This included setting the sensor fusion mode to Nine Degrees of Freedom (NDOF) (which enables quaternion recording) and configuring the accelerometer with a `100.0 Hz` Output Data Rate (ODR) and `16.0g` range. 

An oversight in this project revolved around running both pose and IMU collection concurrently. The `main()` function calls for the initalization of hand pose and then calls for data collection. IMU collection was not called through the main function but rather was initialized earlier due to it having very specific hardware configurations. I attempted to call the function at the same instance as the pose and collection functions, however there was an error in that the laptop used for the experiment would fail to detect the device via BlueTooth.  Therefore the script produced some asynchronous behaviour which was later corrected in the preprocessing stage. The room for improvement of this issue within this stage would call for a further study of the `MetaWear` library and functions which was not explored in this project.

#### Gesture Selection and Relevance

Seven gestures were initially considered for the experiment, each chosen to explore the feasibility of recognizing gestures with varying degrees of complexity and surgical relevance. The gestures were:

1. **Twist Hand**: Rotating the wrist to face the palm towards and away from the camera.
2. **Open-Close Hand**: Opening and closing the hand while the palm faces the camera.
3. **Insert RTL**: Moving a pen horizontally from the left to the right side of the camera view and back.
4. **Insert FTB**: Moving a pen towards and away from the camera.
5. **Twist Pen**: Holding a pen and rotating it 90 degrees and back.
6. **Rotating**: Holding a pen and rotating it in a circular arc.
7. **Lifting Hand**: Lifting a flat hand from a table and putting it back.

Gestures like "Insert RTL", "Insert FTB", and "Twist Pen" are directly inspired by movements commonly made during surgeries, whereas "Twist Hand" and "Open-Close Hand" were included as simpler gestures for proof of concept. "Rotating" and "Lifting Hand" were not further processed due to various constraints, including accidental changes in the laptop webcam position and other minor issues related to the experimental process during the day of recording (in the case of "Rotating", the gesture was performed without breaks in between, making segmentation infeasible).

#### Data Recording and Outputs

The script was executed by myself, performing each selected gesture slowly and accurately around 10 times with a 2 second pause in between each gesture. The camera angle remained consistent (laptop positioned at 90 degrees) throughout the recording to maintain experimental validity. During the data collection, a video was recorded and saved as an MP4 file, which later served as the ground truth for the experiment as the video featured a HUD in the top left corner showcasing the exact system time during the recording of pose data. In addition, the hand pose landmarks and IMU data were saved into separate .CSV files for simplicity and flexibility in preprocessing.

#### Some Constraints

The experiment was conducted in an environment with minimal background noise to ensure data quality. This will rarely be feasible in a surgical environment which is likely be subject to issues involving background clutter, changes in illumination, rapid movement or occulsion[ Paper: Gesture recognition algorithm based on multi-scale feature fusion in RGB-D images]. Another notable constraint during my experiment was an oversight in limiting the MediaPipe function to detect a single hand for capturing hand pose landmarks. As a result, the script recorded columns for two hands, despite only one hand being in view for all experiments. This was later accounted for in the Preprocessing stage. Lastly, the `output.mp4` for all data recording sessions were sped up (approximately 2x). The reason for this issue was not explored in this project although may involve the framerate of the laptop webcam used being 30FPS.

- - - 
## Data Preprocessing

The preprocessing stage for this project involved these two Python scripts: `Align_&_Parse.py` and `Segmentation.py`.  These were designed in order to organize the data into a more usable format for further analysis. They were also used to account for inconsistencies in the recorded data such as the afformentioned IMU sampling executing a few seconds earlier than the pose as well as outliers in the IMU timestamps, notably in the beginning of the data recording. 

#### Align & Parse.py

The `Align_&_Parse.py` script served two purposes: 1) aligning the hand landmark data with the IMU data and 2) parsing the landmarks into a more suitable format for analysis. The script employs a method similar to nearest neighbor interpolation to align the two data streams by their timestamps using the `merge_asof` function from Pandas library. This was needed because the IMU and hand pose data were sampled at different rates. The script also uses regular expressions to extract the x, y, and z coordinates from the string format used in MediaPipe's framework for saving pose data. Parsing these landmarks is also vital for future analysis such as deriving features that will later be used in the machine learning models. The output of this script saves a DataFrame in a .CSV file that will be used for those purposes.

#### Segmentation.py: 

`Segmentations.py`  is a straightforward script that is used on the output dataframe from `Align_&_Parse.py`. It manually segments the dataframe into chunks based on specified start and end index values (cross-referenced with the Timestamp value within the same row). The video output from the recording was used as a ground truth to cross-verify the timestamps in which a gesture or an intermission was performed, which was then manually changed each time before executing the script in a sequential manner. Due to the project having a small dataset, manual segmentation was a convenient and practical choice for accuracy. 

- - - 
## Visualization for Verification

##### Quat_Anim_Vis.py 

In this script, a dataframe containing aligned IMU and pose data is read e.g. `!twisthand_dataframe.csv`. A class called `Quaternion` is used to manage the arithmetic neede for the visualization. In this case, it converts quaternions to rotation matrices and performs quaternion multiplications to compose rotations. A class called `CubeAxes` is then used to create a coloured 3D cube, which updates it's orientation by applying the rotation matrix dervied from the quaternion data. Next, a class called `CubeAxesAuto` is defined which includes animation capabilities provided by matplotlib and updates the orientation frame by frame (per row of dataframe). Finally, a Play/Pause button and progress bar showcasing the current frame are added for convenience. 

##### Quat_Liv_Vis.py 

The prupose of this script is for real time debugging and validation. It requires access to an IMU sensor and utilizes the same connection configurations as the `Multimodal_Data_Recording.py`.  The arriving quaternion data is stored in a buffer ensuring that only recent data is used for visualization. This quaternion data from the buffer is then converted into a rotation matrix and visualized similar to the animated script. However, the approach to `draw_cube` is different than `CubeAxes`. At the time, there was an issue in displaying the colour coded faces of the animated visualization within a live matplotlib environment. Instead colour coded vertices are used, which are not as visually clear. This issue could not be further analysed due to limited access of the IMU device during the time in which this project was develoepd and thus has not been corrected.

##### Handpose_Anim_Vis.py 

The hand pose visualization leverages the framework provided by MediaPipe to define connections between hand landmarks. The landmarks are depicted as red marks, while the connections between them are represented by blue lines. Also using Matplotlib, a 3D scatter plot was initialized to display these landmarks. As the animation progresses, the scatter plot and the connections are updated frame-by-frame, which provides an intuitive way to view the changes in hand pose over time, very similar to the `Quat_Anim_Vis`. To add, a Play/Pause button and progres sslider were also added to the animation, however there was an issue with the animation failing to start in a paused state for all pose related animations. This issue was not explored further within this project.

##### Multimodal_Anim_Vis.py 

The multimodal script essentially combines the features of both the quaternion and hand pose visualizations into a single frame, using a split-screen view. On one side, the hand pose is visualized with landmarks and connections, exactly as in the previous hand pose visualization. On the other side, a 3D cube represents the quaternion rotation, just like the quaternion animated visualization. The same UI features such as the Play/Pause button and progress slider is included, although the issue with the animation failing to start at a paused state is also carried over. This script was used for data scrutiny by comparing the animated visualization with the ground truth video, confirming data integrity.

- - - 
## Feature Selection and Engineering

Feature engineering is essential in this project due to the focus on developing machine learning models. The curation of handcrafted features in this porject not only aim to capture the complex and nuanced information recorded in gesture data but also aim to provide an interpretable framework for understanding surgical gestures. 

The execution and development for extracting each feature set will not be discussed in this document. For further details, please manually inspect each feature extraction script and read the comments associated with the extraction method.

#### Rationale for Feature Categories

##### Statistical Features

**Importance in Data Science**:  
Statistical features was the first approach used in the feature engineering process due to it's commonality across various machine learning disciplines, not just gestural analysis. The reason for this comes from their ability to capture the central tendency, dispersion, and shape of the data distribution, which provides an informative, snapshot of the data characteristics. [Insert Lit Example- https://medium.com/analytics-vidhya/statistics-mean-median-mode-variance-standard-deviation-47fab926465a]

**Features Employed and Justification**:  
This project utilized the measures `mean, median, range, standard deviation, skewness, and kurtosis`. The mean and median provide a summary measure that describe the "central location" of the data, essential in capturing a baseline for each gesture. The standard deviation and range measure the dispersion, skewness captures any asymmetry, and kurtosis identifies the "tailedness" of the distribution. The overview generated by extracting these features were critical for differentiating gestures that may have similar central tendencies but differ in other statistical aspects.

##### Time-Domain Features

**Importance in Data Science**:  
Time-domain features were also essential in this project due to it's importance in applications requiring temporal analysis, such as speech recognition, signal processing, and in this case, gesture recognition [Insert Lit Example - https://revistia.org/files/articles/ejis_v2_i3s_16/Cemil.pdf]. These features capture essential characteristics of time-series data, such as volatility, magnitude, and periodicity, which are vital for distinguishing complex human activities like gestures.

**Features Employed and Justification**:  
This project utilized the measures  `mean absolute difference, rms, peak count, and mean time between peaks`. The mean absolute difference captures the average magnitude of changes in the signal, thus providing insights into the overall volatility or stability of hand movements. The root mean square is a standard method of providing a robust measure of signal magnitude. Peak count identifies the number of significant peaks in the waveform, important for recognizing repetitive or unique movements within a gesture and mean time between peaks captures the average interval between significant peaks thus providing insight into periodicity of certain gestures. 

##### Frequency-Domain Features

**Importance in Data Science**:  
Frequency-domain features provide important analysis of time series data when there are periodic or cyclical aspects to the data[Insert Lit Example - https://link.springer.com/chapter/10.1007/978-3-319-63558-3_40]. Due to the fact that each gesture was performed subsequently, transforming this data into the frequency spectrum reveales these aspects in ways that aren't inherently discernable in the time domain. Extracting these features successfully aids in accurate classification of gestures.

**Features Employed and Justification**:  
This project utilized the measures `dominant frequency, low energy, mid energy, high energy`. Dominant Frequency identifies the frequency component with the highest amplitude within the spectrum, with the intention of identifying cyclical movement such as a hand twist. Low Energy identifies slow and deliberate movements within gestures, with Mid and High Energy being used to capture faster and more rapid gesture movements. The bands chosen in this project were experimental and not based on any existing literature or approaches, thus having much room for optimization. 

##### Landmark-Based Features

**Importance in Data Science**:  
While the other features extracted deal with the temporal and spectral aspects of gesture data, Landmark-based features are unique in dealing with the spatial and geometric aspects of gesture data[Insert Lit Example - https://www.sciencedirect.com/science/article/pii/S1389041720300206?casa_token=EZFCnpBIcTkAAAAA:g85e_kb-L9ZxNhjvzvYY02vuQOtzY7maIq5CczR1uEMAMcu-VpL-Ch2_4mOTpikWmKXpVxlnyw]. These features are highly interpretable and offer a level of granularity that isn't feasible with other feature sets, making them indispensable in machine learning modelling for gesture analysis.

**Features Employed and Justification**:  
This project utilized the measrues `Mean distance between landmarks 4 to 8, Standard deviation of distance between landmarks 4 to 8, Maximum distance between landmarks 4 to 8, Minimum distance between landmarks 4 to 8, Median distance distance between landmarks 4 to 8, Peak distance between landmarks 4 to 8, Same measures but for Landmarks 4 to 12, 4 to 16, 4 to 20, 0 to 4, 0 to 8, 0 to 12, 0 to 16, 0 to 20, 8 to 12, 12 to 16, 16 to 20. >Mean, Std, min, max, median & peaks for specific angles each: thumb, wrist, index. Angle 2: wrist, middleMCP, middleTip. Thumb curvature, Index curvature, middle curvature, ring curvature, pinky curvature, hand curvature. Speed, Direction, Acceleration, Jerk`

This feature set is the most extensive and contextually relevant, in comparison to the other the sets of features extracted due to the importance that geometric relationships between landmarks has on gesture recognition. For instance, the distance between the thumb(`landmark 4`) and index finger (`landmark 8`) could be indicative of certain types of gestures, such as pinching or pointing (which would encompass all pen related gestures within this project). Similar rationale was used for the selection of other `distance` features such as Thumb and Pinky, which provides information on the spread of the hand.  The angles between certain landmarks could also be informative. For example, the angle subtended between the thumb `0`, middle finger MCP `9` and middle finger TIP `12` could capture how bent the hand is (useful for "Open-Close" within this project). Curvature related features intend to extract similarly distinct information such as to see whether the hand is bent or maintaining a straight position. The set of features relating to speed and it's derivatives were curated to get a sense for how fast or slow a gesture is, although these features in particular are vulnerable to high correlation within this feature set, which is later considered. 

##### Acceleration Magnitude

**Features Employed and Justification**:  
This project utilized the measures `mean, std, max, min, median` 
Acceleration magnitude serves as an effective proxy for the speed and suddenness of movements, crucial for understanding the dynamic aspects of surgical gestures. The statistical features extracted served a purpose similar to statistical features extracted in the other feature set.

### Feature Selection and Removal

##### Prior Removal

Several features were removed prior to conducting further analysis on the dataset. In particular `mean time between peaks` was removed due to having NaN values making the data more difficult to work with in modelling. Furthermore `mid energy, high energy` were removed due to all feature values equating to zero throughout all five gestures and intermission data within the dataset, imploring that this feature should be tuned to much smaller frequency values for gestural analysis in surgical context, although the extent of this goes beyond the scope of this project. Finally, `dominant frequency` was also removed for similar reasons as most feature values equated to zero across the gestures. `Feature_Removal.py` was used to remove these features and subsequently deduced features to remove.

##### Correlation Analysis

Correlation analysis was performed to identify highly interrelated features that could potentially add redundancy to the machine learning models. The first script developed was `correlated_features.py` which defined a simple function to identify features correlated by over 90%. However, the findings from this script were not utilized due to a lack of accounting for sequentially related data. For example, `stat_mean_z_1` and `stat_mean_z_2` had a correlation rate over 0.99, however that did not mean one or the other should be removed due to the importance that sequential data has in contextually understanding the gestures. Therefore, more refined scripts had to be developed which skipped redundant comparisons. Using these new scripts `statistical_correlations.py`, `timedomain_correlations.py`, `frequency_correlations.py` and `landmark_correlations.py`, the findings below were examined.

In the statistical feature set, `stat_mean` and `stat_median` were found to be highly correlated with a correlation coefficient exceeding 0.9. Given this high correlation, `stat_median` was removed to reduce multicollinearity. Similarly, `stat_std` and `stat_range` also exhibited a high correlation, leading to the removal of `stat_range`. For time-domain features, no pair had a correlation above 0.6, so all remaining features were retained. In the case of landmarks, mean and median values for distance, angle, and curvature were often highly correlated; `median` was retained due to its robustness to outliers. Additionally, `max` and `min` distances were highly correlated with `std`, but only `max` and `min` were removed, keeping `std` for its potential to capture the variability in the data. Lastly, in the acceleration magnitude feature set, `acc_mag_std` and `acc_mag_max` showed significant correlation, resulting in the removal of `max` and `min`. 

##### Heatmaps

Note To Self: Talk about the execution and development of the script.

Heatmaps were generated to offer a visual representation of how each feature correlates with others. The 21 landmarks were aggregated so that comparison with quaternion and acceleration features was feasible. 

[Insert Pic Here, maybe 1 or 2 pics.]

However, dimensionality reduction was not performed as a result of any analysis from the heatmaps. For the interest of future work in this avenue, here is a list of information i deduced specific to this project and it's dataset for highly discriminative features as a result of my subjective viewing of each heatmap script.

	fr_low_energy for ax,ay,az 
	stat_skewness 
	stat_kurtosis for ax,ay,az
	td_mean_diff except quaternion features (w,x,y,z)
	td_peak_count for landmarks x,y,z
	lm_distance_max for 4_8,4_12,4_16 & 4_20.
	lm_distance_min for 0_4,0_8,0_12,0_16 & 0_20
	lm_distance_num_peaks 
	lm_angle_mean for wrist_middleMCP_middleTip 
	lm_angle_min for wrist_MiddleMCP_middleTip
	lm_angle_num_peaks for thumb_wrist_index and wrist_MiddleMCP_middleTip 
	lm_curavture_mean for index, middle, ring and pinky
	lm_curvature_median for index, middle, ring and pinky
	lm_speed_mean for x and y 
	lm_direction_mean for z 
	lm_direction_max/min/std for x,y,z
	lm_direction_num_dir_change for y 
	lm_accel_median for x,y,z
`
##### Dataframe Compilation for Modelling

Following feature extraction and selection, a comprehensive dataframe was created using `Making_Modelling_Dataframe.py` by aggregating individual feature segments from all gestures and intermissions. In this project, there were 10 specific gesture segments and two intermission segments per gesture. This script was actually developed prior to feature selection and thus the `Feature_Removal.py` script was used on the outputted dataframe in an iterative manner.

##### Normalization 

Once all feature engineering was concluded, the dataframe created was normalized to bring all values within each feature set in a comparable range. This is necessarily as most machine learning algorithms are sensitive to feature scales, like k-NN and SVM for example. Min-Max scaling was employed for this purpose using the `MinMaxScaler` from the `sklearn.preprocessing` library. All the feature columns underwent this transformation, while label columns like `Gesture Name` and `Gesture Type` were excluded from scaling. After normalization, the DataFrame was saved for subsequent use in machine learning models.

##### Multi-Dimensional Scaling (MDS)

`MDS_visualization.py` was a useful script that visualized the high-dimensional feature space in a 2D format. This helped to provide an intuitive understanding of how well the features cluster different gestures, serving as a preliminary validation of the feature set's effectiveness. [Insert Lit Example - https://cannoodt.dev/2019/11/lmds-landmark-multi-dimensional-scaling/]

[Insert Picture Here]

The results showed surprisingly good clustering, with close overlap appearing in certain instances between `InsertFTB` and the `Twisthand` gesture (more prevalent when the `random_state` is modified). A possible reason for this may be due to these two gestures having the most movement in the Z axis for pose data, due to being prone to move toward and away from the camera. This could call for a linear classifier being used when working with small gesture datasets like within this project, however that is unlikely to scale with size.

##### Machine Learning Techniques for Validation

Insights from the MDS visualization called for more feature selection. Random Forest was explored as the tool to rank feature importance. [Insert Lit Example - https://www.kaggle.com/code/prashant111/random-forest-classifier-feature-importance]. Using the `RF_FeatureImportance.py` script, a stratified split of 70% training and 30% testing was executed on the normalized dataset. To ascertain the robustness and consistency of feature importance rankings, the Random Forest model was run 30 times with varying random states. The features consistently appearing in the top 20 rankings across all iterations were noted below, offering a measure of reliability in their discriminative power. However, due to achieving reasonable clustering with MDS and valid results in later modelling stages, further feature selection was not pursued.

[Insert Pic of Results Here]

Further validation was achieved through clustering metrics. In particualr, the Silhouette and Davies-Bouldin Scores[Insert Lit Example - https://towardsdatascience.com/performance-metrics-in-machine-learning-part-3-clustering-d69550662dc6] were explored. These metrics were calculated using K-means clustering on the normalized feature set. The Silhouette Score offered insight into how similar each object is to its own cluster compared to other clusters, providing a measure of how well separated the gesture types are in the feature space. The Davies-Bouldin Score complemented this by evaluating the average similarity ratio of each cluster with its most similar cluster; lower values indicate better clustering. These clustering metrics served as a quantitative means to validate the effectiveness of the selected features, thus providing a fuller, more nuanced understanding of their discriminative capabilities.

[Insert Pic of Results Here]

- - - 
## Machine Learning Models


The final objective in this project was to leverage the curated features for the effective classification and clustering of surgical hand gestures. Below is an outline of the machine learning algorithms employed, their parameter tuning, validation techniques, and results. The choice of algorithms and procedures used follow standard conventions in machine learning detailed below. For ease of comparison between algorithms, accuracy was the chosen metric discussed for model performance.

#### RandomForest.py

1. **Model Description**:  
    RandomForest is an ensemble learning method that fits a number of decision trees on various sub-samples of the dataset. It offers robustness against overfitting and is generally useful for high-dimensional data[Insert Lit Example - https://www.sciencedirect.com/science/article/pii/S2405896319307591], which aligns with the complex feature set utilized in this project.
    
2. **Hyperparameters**:  
    A stratified 70-30 train test split is used in order to maintain class representation (42 training samples, 18 test samples). The model was initiated with 100 estimators (default value) and a random state of 33 for reproducibility. The effect of these hyperparameters on the model's performance was not evaluated within this project. 

[Insert RF Pic here]

3. **Performance Metrics**:  
    The model achieved an accuracy of approximately `0.89`, with notable precision and recall scores for most gestures. However, the "Intermission" category had a lower recall. This is likely due to the smaller sample size for each intermission segment. Although there were 10 intermission segments, the sample size varied drastically depending on the recording. This project is not particularly concerned with categorizing when a gesture is not being performed, however for this to be explored further, a proper dataset of structured "Intermission" gestures may be worth consideration.
    
4. **Cross-Validation**:  
    k-fold Cross-Validation is a standard procedure in machine learning[Insert Lit Example - https://machinelearningmastery.com/k-fold-cross-validation/]. A 5-fold cross-validation was performed, yielding a mean score of 0.95. This high score suggests that the model is robust, but given the small dataset, some caution is needed in generalizing these results.
    

#### k-NN.py

1. **Model Description**:  
    k-Nearest Neighbors (k-NN) is an algorithm that classifies a given data point based on the majority class of its 'k' nearest neighbors. It's prescence in Machine Learning is often in [Insert Lit Example - https://www.researchgate.net/publication/343988404_Smart_Hand_Gestures_Recognition_Using_K-NN_based_Algorithm_for_Video_Annotation_Purposes] however it was explored for it's popularity depsite it's struggle with high-dimensional data.
    
2. **Hyperparameters**:  
    The same statified 70-30 train test split is used, as is the case for all machine learning models explored in this porject. The model also used 4 neighbors for classification. Given the complexity of the feature space, the choice of `k` may requires further optimization in future projects.

[Insert RF Pic here]

3. **Performance Metrics**:  
    The k-NN model also achieved an accuracy of `0.89`, the exact same as the RandomForest model, and again faltered on the "Intermission" category. This adds validity to the patterns of data recognized by the model in that the results are not artifacts of any particular alogrithmic approach. However, the fact that both are achieving the same results also calls into question the complexity necessary for classification of gestures of this dataset. It was deduced earlier in the project that linear classification would not be explored seeing as gesture data is complex, however for such a small dataset, it may achieve similar values as the more complicated  
4. **Cross-Validation**:  
	A 5-fold cross-validation was performed, yielding a mean score of `0.88`. This high score suggests that the model is robust, but given the small dataset, caution is needed in generalizing these results. 

#### SVM.py

1. **Model Description**:  
    Support Vector Machines (SVM) are known for its effectiveness in high-dimensional spaces and its ability to produce hyperplanes that best separate different classes. [Insert Lit Example - https://dspace.mit.edu/handle/1721.1/121856]
2. **Hyperparameters**:  
    An RBF (Radial Basis Function) kernel was used, a common choice for non-linear data and was not tuned in this project.

[Insert RF Pic here]

3. **Performance Metrics**:  
    The SVM model achieved perfect accuracy of `1.00`. This raises concerns about potential overfitting, especially given the limited dataset, and could be an explanation for high accuracy for other model outputs as well.
4. **Cross-Validation**:  
	A 5-fold cross-validation was performed, yielding a mean score of `1.00`. The perfect score for this gives more evidence to suggest that the model results using this alogrithm are dubious.

#### GradientBoosting.py

1. **Model Description**:  
    Gradient Boosting constructs a strong predictive model by optimizing the shortcomings of a collection of weaker models. It's computationally more expensive but is known to yields better results.[Insert Lit Example - https://www.researchgate.net/publication/337446028_Design_of_a_Wearable_Smart_sEMG_Recorder_Integrated_Gradient_Boosting_Decision_Tree_based_Hand_Gesture_Recognition]
2. **Hyperparameters**:  
    The model was implemented with default hyperparameters.

[Insert RF Pic here]
   
3. **Performance Metrics**:  
    Like its counterparts, the Gradient Boosting model also reached an accuracy of `0.89`. However, it managed to perfectly classify the "Open Close" gesture, unlike k-NN which perfectly classified "Twist Hand".
4. **Cross-Validation**:  
	A 5-fold cross-validation was performed, yielding a mean score of `0.92`. This high score suggests that the model is robust, but given the small dataset, some caution is needed in generalizing these results. 


More tuning, analysis and development of advanced models such as Neural Networks utilizing PyTorch were considered, however due to time constraints and the limitations imposed by the small dataset collected, these were not explored. 

- - - 
## Scalograms 

The final aspect of this project briefly explores the generation of scalograms. The choice of scales used was Continuous Wavelet Transform (CWT) due to being ideal for analyzing nonstationary signals, which encompasses gesture recognition[Insert Lit Example - https://uk.mathworks.com/videos/deep-learning-for-engineers-part-4-using-transfer-learning-1617282574525.html].  Below is an output of the scalogram generated for IMU data for the "Twist Hand" gesture:

[Insert Pic Here]

The results are informative as the first section (approx 0-4000) of the sample count lacks any notable shape and is darker, which represents the "Intermission" after being compared with the ground truth. The rest of the sample count represents a gesture being performed as it shows more vivid colours and spikes indicating that these are higher frequency than the section prior. Within this gesture, it is clear that `x`,`y`,`z`,`ax` and `az` are the most informative for gesture recognition. 

There are many avenues to explore with scalograms, and while research has been done in using scalograms ot generate 2D CNN's [Insert Lit Example - https://academic.oup.com/jcde/article/10/4/1694/7218563], there is few literature focused on Multimodal aspects. Within the field of Surgical Data Science, scalograms could serve as a foundation for fusing time and frequency information from different modalities like video, audio, IMU, pose, etc in order to gain a more holistic understanding of surgical gestures. With a rich, fused dataset in a unified time-frequency domain, advanced machine learning models could be trained to recognize more complex patterns that wouldn't be apparent in any single modality. By fusing data from different sources, it may be possible to not only recognize a particular gesture but also understand its context. For instance, a certain hand movement could mean different things depending on the spoken commands or the other tools being used, information which might be captured in the audio or video data. Therefore scalograms would be a great addition to the work of the Surgical Data Science Team. Lastly, Transfer Learning may also want to be explored as an alternative to curating an image classifier from scratch which is a challenging task in itself [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8840381/#B19-sensors-22-00706] 


--- 
## Limitations and Future Work

* There is a complete lack of expert validation providing insights into gestures that are most relevant in the surgical section, or more specifically for Transpphenodial surgeries. Future work on a multimodal dataset needs this so that certain angles or movements more indicative of srugical actions can be understood and discernable, increasing feature robustnes. This adds quality assurance that goes beyond what a model is capable of currently.
* This project is not inherently scalable with the libraries and scripts developed. For example, the approach taken in visualization with `matplotlib` likely would not work with a larger, more extensive dataset. Not only would it perform poorly computationally due to the sheer size of the data, a lack of useful tools like being able to rewind and fast forward through an animation of the data would require tools that go beyond the `widget`  subset library of `matplotlib`.
* The project did not explore the implications of NaN data, limiting the generalizability of findings and scalability as NaN data is commonplace in larger datasets. 
* The use of `merge_asof` for nearest neighbor interpolation has worked for this project but may not be the most accurate way to handle larger datasets. This is because the `system time` used to align both data streams was not preprocessed prior to using this interpolation. A system like [Insert Lit Example - https://www.stereolabs.com/docs/sensors/time-synchronization/#:~:text=Accurate%20time%20synchronization%20is%20required,Epoch%20time%20and%20nanosecond%20resolution.] should be considered to ensure that the timestamps for both streams of data are as accurate as possible before using the function.
* Manual segmentation using ground truth verification is susceptible to human error, introducing a level of variablity into the dataset that affects relability of data curation. It is also a time consuming process that is not scalable. Automatic segmentation should be considered [insert Lit Example - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7825741/]
* Curation of hand crafted features in this project has a lot of room for improvement that can only be validated by more extensive testing. i.e. these features may have performed well in very specific environments whereas performance may degrade if the gesture collection is subject to more varied conditions. Automatic Feature learning using architectures like CNNs are standard in research currently [Insert Lit Example - https://www.mdpi.com/2504-2289/7/2/102] and should be used. 
* Anomaly detection could be explored within multimodal gestural analysis. The question of whether a gesture is interrupted could serve important value in advancing the field of surgical data science. In particular, [Insert Lit Example - https://ieeexplore.ieee.org/abstract/document/9812034?casa_token=rQ8s5ahrC8sAAAAA:ZAy-wMMkFg6p1Cx8eyBWJbtfNO0LsNlPkHFFl3uYLBaDKGPCjqMNAoqnCs7C1rSEcWXhRsIH].
* Future work using advanced modelling should explore Temporal Sequence analysis. Basically if a gesture occurs one after the other, we can comment on them, pick them out etc. Using Hidden Markov Models, LSTMs, Transformers and so on offer advanced tools to creating more context aware gesture recognition [Insert Lit Example - https://napier-repository.worktribe.com/output/2816883/dynamic-hand-gesture-recognition-using-3d-cnn-and-lstm-networks]. These are only feasible with larger datasets than this project.
* Multimodal Feature Fusion will be the standard taken for future endeavours in Surgical Data Science. While this project explores pose and IMU, it goes without saying that the integration of more modalities aids to the robustness and contextual awarness of gesture recognition models in the future. Limited research exists currently exploring this for gestural analysis however some approaches may be worth considering here [Insert Lit Example - https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12712]
* Real-time analysis will be a crucial aspect of Surgical Data Science in the future due to the ability to apply immediate feedback, account for irregularities and errors in gesture patterns, training and so on. Expanding the models developed within this project for real time purposes is an important step forward iNto making this avenue a reality.




A good time to mention TriBERT likely...

Room for transformers or suttin(title: HGR-ViT: Hand Gesture Recognition with Vision Transformer ): Hand gesture recognition (HGR) is a crucial area of research that enhances communication by overcoming language barriers and facilitating human-computer interaction. Although previous works in HGR have employed deep neural networks, they fail to encode the orientation and position of the hand in the image. To address this issue, this paper proposes HGR-ViT, a Vision Transformer (ViT) model with an attention mechanism for hand gesture recognition. Given a hand gesture image, it is first split into fixed size patches. Positional embedding is added to these embeddings to form learnable vectors that capture the positional information of the hand patches. The resulting sequence of vectors are then served as the input to a standard Transformer encoder to obtain the hand gesture representation. A multilayer perceptron head is added to the output of the encoder to classify the hand gesture to the correct class. The proposed HGR-ViT obtains an accuracy of 99.98%, 99.36% and 99.85% for the American Sign Language (ASL) dataset, ASL with Digits dataset, and National University of Singapore (NUS) hand gesture dataset, respectively. 


This paper 100% should be used in discussing automatic feature extraction (https://ieeexplore.ieee.org/abstract/document/8215729?casa_token=s_DFmBHyZXYAAAAA:YjuRA73qhHpUhorlgO89KQECRIVt65QZNPMRRcKXx9kLgXk0TlNLDUgC251ddvuCJr5z7f08)

 
- - - 
## List of References




- - -

