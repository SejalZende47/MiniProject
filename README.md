# MiniProject
"A Human Activity Recognition (HAR) system using smartphone sensors and ML-based context detection, optimized for efficient mobile deployment."

📌 Project Overview 
This project focuses on identifying human physical activities (such as walking, sitting, or climbing stairs) using data captured from smartphone-embedded Tri-axial Accelerometers and Gyroscopes. The primary goal is to develop a lightweight machine learning model capable of Efficient Mobile Deployment with low computational overhead.

🚀 Key Features
Sensor Fusion: Combines 3-axis linear acceleration and 3-axis angular velocity.
Feature Engineering: Implements time-domain and frequency-domain feature extraction.
Efficiency-First: Optimized for mobile environments using feature selection to reduce battery consumption and latency.
Context Awareness: Real-time classification of user activities to trigger context-aware mobile responses.

🛠️ Tech Stack
Language: Python 3.x
Libraries: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
Deployment Target: Mobile (TFLite / ONNX optimized)

📊 Dataset Information
The project utilizes the UCI HAR Dataset (or your custom collected data).
Activities: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying.
Sensor Frequency: 50Hz.

📁 Repository Structure
├── data/               # Sample sensor datasets (CSV)
├── notebooks/          # Data exploration and Model training
├── models/             # Saved model files (.pkl, .tflite)
├── scripts/            # Preprocessing and Feature extraction scripts
├── requirements.txt    # List of required Python libraries
└── README.md           # Project documentation

📈 Future Scope
Implementation of CNN-LSTM architectures for improved temporal pattern recognition.
Full integration with an Android/iOS application for real-time inference.
Energy-efficiency benchmarking for edge-device deployment.
