# Multimodal-Gestural-Analysis
Multimodal Gestural Analysis during Surgical Interventions

## Introduction:

This project was developed during my summer internship with the goal of classifying gestures in a surgical context using a multimodal data-driven approach. I specifically explored the use of hand pose landmarks using MediaPipe’s framework and IMU data in the form of quaternions and acceleration using an Mbientlab wearable wrist sensor. 

During my time, I devised a short set of experiments in which I performed several hand gestures varying in levels of surgical relation. I then used this recorded data to develop a pipeline that integrated these diverse datatypes into a structured dataset. From there, numerous features were extracted in order to encapsulate various spatial, temporal, and statistical characteristics of hand movement such as speed, acceleration magnitude and geometric relationships between landmarks. These features served as the input for machine learning models, which were then trained to recognize and classify the different gestures. I also explored the use of data visualization as a tool to scrutinize the data that was recorded but also for feature selection. While the dataset and models developed are preliminary in size, setup, and feature set, I believe they offer a good starting point for more extensive work in the future applying AI and data science in surgery.

All scripts were developed using Python with some prominent libraries in this project being pandas, sklearn, matplotlib and seaborn.

## Setup 

#### Hardware Requirements:

1. **Computer with Webcam and Bluetooth**: For capturing hand pose landmarks.
2. **Mbientlab Wearable Wrist Sensor**: Required for IMU data collection.
3. **Optional**: RealSense Depth Camera for more accurate hand landmark data. Replace your webcam with this for enhanced data quality.

#### Software Requirements:

- Python (used during project = 3.10.9)
- IDE (used during project = Visual Studio Code)
  
#### Installing Dependencies:

You'll need to install several Python libraries to run the scripts in this repository. Below are the libraries used across different parts of the project:

- `os`, `time`, `datetime` for file and time operations.
- `metawear` for interfacing with the Mbientlab wearable wrist sensor.
- `cv2` and `mediapipe` for video capture and hand pose estimation.
- `pandas` and `numpy` for data manipulation.
- `matplotlib` and `seaborn` for data visualization.
- `scipy` for signal processing.
- `sklearn` for machine learning tasks.
- `pywt` for wavelet transforms.

After cloning the repository in your local environment, run the following command in your terminal to install all dependencies:

```bash
pip install -r dependencies.txt
```

##### Note:

1. **Mbient Sensor Connectivity**: The MAC address for the Mbient sensor is hardcoded in the script and may be subject to change. If the sensor disconnects frequently, press its button once or twice before running the script to ensure it remains connected.
2. **Path Customization**: Most scripts contain hardcoded paths. Make sure to modify them to fit your directory structure.

---
