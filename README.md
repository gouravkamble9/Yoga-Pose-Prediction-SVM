# Yoga Pose Classification using SVM


This Python project aims to classify yoga poses using a Support Vector Machine (SVM) algorithm. The project utilizes various libraries such as Mediapipe, OpenCV (cv2), NumPy, Pandas, scikit-learn, and pickle to extract body angles from a dataset, train a classification model, and perform live predictions of yoga poses.

## Project Description

Yoga is a popular practice that combines physical postures, breathing exercises, and meditation techniques to improve physical strength, flexibility, and mental well-being. This project focuses on using computer vision techniques and machine learning to classify different yoga poses automatically.

The project consists of the following main steps:

- Data Collection: A dataset containing images or videos of 18 different yoga poses is gathered. Each pose should have multiple samples from different angles and perspectives.

- Pose Estimation: Mediapipe library is used to estimate the body pose in each image or frame of a video. This involves detecting and tracking key points on the body, such as joints and landmarks.

- Angle Extraction: Using the key points obtained from pose estimation, the project calculates six important angles for each pose. These angles represent the body's posture and are crucial for distinguishing between different poses.

- Data Preprocessing: The extracted angles are organized into a structured format suitable for training a machine learning model. The data is stored in a CSV (Comma Separated Values) file, which can be easily read and processed by other libraries.

- Model Training: The scikit-learn library is utilized to train a Support Vector Machine (SVM) model on the preprocessed data. The SVM algorithm is chosen for its effectiveness in classification tasks, especially when dealing with high-dimensional feature vectors.

- Model Evaluation: The trained SVM model's performance is assessed using appropriate evaluation metrics such as accuracy, precision, recall, and F1 score. This ensures that the model has learned to classify yoga poses accurately.

- Model Saving: Once the SVM model is trained and evaluated, it is serialized and saved to disk using the pickle library. This allows the model to be reused without retraining in future sessions.

- Live Pose Classification: Using a webcam or live video feed, the project performs real-time pose estimation and feeds the extracted angles into the trained SVM model for classification. The predicted yoga pose along with the probability is displayed on the screen.

### Dependencies

The project relies on several Python libraries, which need to be installed prior to running the code:

- mediapipe
- cv2 (OpenCV)
- numpy
- pandas
- scikit-learn
- pickle
- math
- os
csv

## File Descriptions

### 1. angle_csv.ipynb
This file contains code for extracting angles from a dataset and saving them to a CSV file. It includes functions to perform pose estimation using Mediapipe, calculate the necessary angles, and store them in a structured format for further processing.

### 2. model.ipynb
The model.ipynb file is responsible for creating the SVM classification model. It utilizes scikit-learn to train the model on the extracted angles from the CSV file. The trained model is then saved using pickle, allowing it to be reused for predictions without retraining.

### 3. prediction.ipynb
In prediction.ipynb, the SVM model created in model.ipynb is loaded, and it performs predictions on test images or a separate dataset. This file uses OpenCV (cv2) to read and process images, and the SVM model to classify the yoga poses. It displays the predicted pose name and the associated probability.

### 4. live.ipynb
The live.v file provides functionality for real-time pose estimation and live prediction of yoga poses. It uses a webcam or live video feed to capture frames, performs pose estimation using Mediapipe, extracts the angles, and feeds them into the SVM model for classification. The predicted pose, along with its probability, is displayed on the screen in real-time.


### Usage

To use this project, follow these steps:

- Prepare your dataset: Collect images of the 18 different yoga poses and organize them in a specific directory structure.
![image](https://github.com/gouravkamble9/Yoga-Pose-Prediction-SVM/assets/61933116/5aaaf8fc-8125-4bfe-a22f-75ced3784a43)

- Run `angle_csv.ipynb`: Execute this file to extract angles from the dataset and save them to a CSV file.
- Run `model.ipynb`: Train the SVM model using the angles extracted in the previous step and save the model using pickle.
- Run `prediction.ipynb`: Use this file to test the trained SVM model on test images or a separate dataset and view the predicted results.
- Run `live.ipynb`: Execute this file to perform real-time pose estimation and live prediction using a webcam or live video feed.

***Note: Make sure to update the file paths and other necessary configurations according to your dataset and requirements.If you don't have a dataset, you can still run the live.py file for live yoga pose prediction using a webcam .***
