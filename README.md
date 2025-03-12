# Alzheimer-detection-and-classification
This repository contains a deep learning-based approach for detecting and classifying Alzheimer's disease using CNNs. The project includes dataset preprocessing, model training, evaluation, and a GUI for user-friendly interaction

🧠 Alzheimer's Detection and Classification
📌 Overview
This project focuses on detecting and classifying Alzheimer's disease using deep learning techniques. A Convolutional Neural Network (CNN) is trained to analyze MRI scans and classify them into different stages of Alzheimer's. The project includes a Graphical User Interface (GUI) for easy interaction.

📂 Dataset
The dataset used for training and evaluation is sourced from Kaggle:
🔗 Best Alzheimer's MRI Dataset - 99% Accuracy

It contains MRI scans categorized into different stages:
✅ Mild Dementia
✅ Moderate Dementia
✅ Non-Demented
✅ Very Mild Dementia

🛠 Technologies Used
Python (TensorFlow, Keras, OpenCV, NumPy, Matplotlib)
MATLAB (GUI Development)
Deep Learning (MobileNetV2 for feature extraction)
🚀 Features
✔️ Load MRI images for classification
✔️ Use a trained CNN model to predict the stage of Alzheimer's
✔️ Train a new model if required
✔️ User-friendly MATLAB-based GUI

📌 Installation & Usage
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/mvnikhitha/Alzheimer-detection-and-classification.git
cd Alzheimer-detection-and-classification
2️⃣ Run the GUI
Open MATLAB and run:

matlab
Copy
Edit
alzheimers_gui
3️⃣ Train the Model (Optional)
If you want to retrain the model, ensure your dataset is available in the correct directory and run:

matlab
Copy
Edit
trainCNNModel()
📊 Model Training Details
Pretrained MobileNetV2 model is used for feature extraction.
The final layers are replaced to classify MRI images into Alzheimer's stages.
Adam optimizer is used for training.
🎯 Results
The model achieves high accuracy on the test dataset, making it a reliable tool for Alzheimer's detection.

📜 Acknowledgment
Special thanks to Luke Chugh for providing the dataset on Kaggle.

