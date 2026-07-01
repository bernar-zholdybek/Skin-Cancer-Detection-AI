# Skin Cancer Analysis Application

A desktop application for skin disease classification using deep learning. Upload an image of a skin lesion and receive an AI-based analysis with the most likely diagnosis and a cancer/non-cancerous risk indication.

Built with Python, TensorFlow, and CustomTkinter.

## SUPPORTED SKIN DISEASE CLASSES

0. Actinic Keratoses and Intraepithelial Carcinoma – Cancerous
1. Basal Cell Carcinoma – Cancerous
2. Benign Keratosis-like Lesions – Non-cancerous
3. Dermatofibroma – Non-cancerous
4. Melanocytic Nevi – Non-cancerous
5. Melanoma – Cancerous
6. Vascular Lesions – Non-cancerous



## Project Structure
skin-cancer-analysis/
│
├── main.py                             # Main application file
├── ham10000_cnn_transfer_learning.h5   # Trained AI model
├── requirements.txt                    # Required dependencies
└── README.md


## Prerequisites
* Python 3.10+
* Trained model file ham10000_cnn_transfer_learning.h5 placed in the same directory as main.py


## Required Libraries

```bash
tensorflow
customtkinter
pillow
numpy
```

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/bernar-zholdybek/Skin-Cancer-Analysis.git
cd Skin-Cancer-Analysis
```

### 2. Dependency Installation
Install all required packages
```bash
pip install -r requirements.txt
```

### 3. How to Run
```bash
python main.py
```
Then upload a skin lesion image and click Analyze with AI to receive results.


## Tech Stack


* **UI:** CustomTkinter
* **AI Framework:** TensorFlow / Keras
* **Base Model:** MobileNetV2 (pre-trained on ImageNet, transfer learning)
* **Image Processing:** Pillow (PIL), NumPy
* **Input:** 224 × 224 RGB images
* **Output:** Top-3 predicted diagnoses with confidence percentages


## AI Model

The classifier uses transfer learning on a frozen MobileNetV2 feature extractor, trained on the HAM10000 real-world dermatoscopy dataset. The model architecture adds a Global Average Pooling layer, fully connected dense layers, a Dropout layer for regularisation, and a Softmax output layer over 7 classes.


## About Application Limits

This application is developed for educational purposes only
It must not be used as a substitute for professional medical diagnosis
Always consult a qualified healthcare professional for medical advice
