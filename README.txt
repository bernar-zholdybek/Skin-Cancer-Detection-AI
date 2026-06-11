Skin Cancer Analysis Application
XMUM AIT102 – Python and TensorFlow Project

==================================================

PROJECT OVERVIEW

This project is a desktop application for skin disease classification using Artificial Intelligence and Deep Learning.

The application allows users to upload an image of a skin lesion and receive an AI-based analysis indicating the most likely diagnosis and whether the condition is cancerous or non-cancerous.

The system is developed using Python, TensorFlow, and a pre-trained MobileNetV2 convolutional neural network, combined with a modern graphical user interface built with CustomTkinter.


==================================================

PROJECT OBJECTIVES

The main objectives of this project are:

- To apply deep learning techniques to real-world medical image classification.
- To build a user-friendly graphical interface for non-technical users.
- To demonstrate the use of transfer learning in computer vision.
- To classify skin lesions into seven clinically relevant categories.
- To provide clear cancer risk indication.


==================================================

AI MODEL DESCRIPTION

Base Model:
MobileNetV2 (pre-trained on ImageNet)

Technique:
Transfer Learning

Input Size:
224 x 224 RGB images

Output:
Probability distribution over 7 skin disease classes

Model Format:
.h5 (Keras)


Model Architecture:

- Frozen MobileNetV2 feature extractor
- Global Average Pooling layer
- Fully connected dense layers
- Dropout layer for regularization
- Softmax output layer


==================================================

SUPPORTED SKIN DISEASE CLASSES

0. Actinic Keratoses and Intraepithelial Carcinoma – Cancerous
1. Basal Cell Carcinoma – Cancerous
2. Benign Keratosis-like Lesions – Non-cancerous
3. Dermatofibroma – Non-cancerous
4. Melanocytic Nevi – Non-cancerous
5. Melanoma – Cancerous
6. Vascular Lesions – Non-cancerous


==================================================

APPLICATION FEATURES

- Modern dark-themed graphical interface
- Image upload support (.jpg, .png, .jpeg)
- Automatic image preprocessing
- AI-based prediction using TensorFlow
- Display of top-3 predicted diagnoses with confidence percentages
- Clear cancer risk indication
- Error handling for missing images or model issues


==================================================

TECHNOLOGIES USED

- Python 3
- TensorFlow / Keras
- NumPy
- Pillow (PIL)
- Tkinter
- CustomTkinter


==================================================

PROJECT STRUCTURE

project/
|
|-- main.py                            Main application file
|-- ham10000_cnn_transfer_learning.h5  Trained AI model
|-- README.txt                         Project documentation


==================================================

HOW TO RUN THE APPLICATION

1. Install required dependencies:

pip install tensorflow customtkinter pillow numpy

2. Ensure the trained model file
   ham10000_cnn_transfer_learning.h5
   is located in the same directory as the Python file.

3. Run the application:

python main.py

4. Upload a skin lesion image and click "Analyze with AI" to receive results.


==================================================

IMPORTANT NOTICE

This application is developed for educational purposes only.
It must not be used as a substitute for professional medical diagnosis.
Always consult a qualified healthcare professional for medical advice.


==================================================

ACADEMIC CONTEXT

This project was developed as part of the AIT102 – Python and TensorFlow course
at Xiamen University Malaysia (XMUM).

The project demonstrates practical skills in:
- Deep Learning
- Computer Vision
- GUI Development
- Model Deployment
