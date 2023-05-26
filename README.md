# Face Data Collection and Recognition

## Description:
This project consists of two Python scripts: FaceData.py and FaceRecognition.py. The first script, FaceData.py, captures face data from a webcam, detects faces in the frames, and saves the collected data as a numpy array. The second script, FaceRecognition.py, uses the collected data to train a K-Nearest Neighbors (KNN) classifier and performs real-time face recognition using the trained model.

## Requirements:
- Python 3.x
- OpenCV (cv2) library
- NumPy library
- scikit-learn library

## Setup:
1. Make sure you have Python 3.x installed on your system.
2. Install the required libraries by running the following command in your terminal:

```
pip install opencv-python numpy scikit-learn
```

## Usage:

1. FaceData.py
- This script collects face data from a webcam and saves it for later use in face recognition.
- Run the script by executing the following command in your terminal:
  ```
  python FaceData.py
  ```
- Enter a name when prompted to label the collected face data.
- The webcam will open, and faces detected in the frames will be displayed in a separate window.
- Press 'c' on your keyboard to capture the face data. The captured face will be shown in grayscale.
- Repeat the capture process for different faces.
- Press 'q' to exit the script. The collected face data will be saved in the 'faces.npy' file.

2. FaceRecognition.py
- This script performs real-time face recognition using the previously collected face data.
- Make sure you have collected face data using the FaceData.py script and have the 'faces.npy' file available.
- Run the script by executing the following command in your terminal:
  ```
  python FaceRecognition.py
  ```
- The webcam will open, and faces detected in the frames will be recognized using the KNN classifier.
- The recognized face label will be displayed on the frame.
- Press 'q' to exit the script.

## Note: 
- The face data collected by FaceData.py will be appended to the existing 'faces.npy' file if it already exists.
- You can customize the size of the captured faces and recognized faces by modifying the 'cv2.resize' function parameters in both scripts.
- Ensure proper lighting and positioning of faces during data collection for better recognition accuracy.

