# Car Brand Recognition System

Submitted by:

* Vanshika Garg
* Agrim Goyal
* Kartikay Chadha
* Aditi Sharma
* Kanish Chadha


This repository contains the source code for a Deep Learning-based Car Brand Recognition project.The system is designed to classify vehicle makes and models from images, addressing the challenge of fine-grained classification where different sub-categories (e.g., specific models or years) can be visually similar.

##  Project Overview

Vehicle Make and Model Recognition  is a specialized subset of object classification.Unlike generic object detection, it requires distinguishing between subtle features.This project leverages Transfer Learning with pre-trained Convolutional Neural Networks (CNNs) to automate this process efficiently.

The project implements and compares two major architectures:
* **ResNet50** (Transfer Learning)
* **InceptionV3** (Feature Extraction)

## 📊 Dataset

The project utilizes the **Car Logo Dataset** sourced from Kaggle.
* **Content:** Full and partial vehicle images organized into class folders representing major manufacturers.
* **Classes:** Approximately 15 distinct classes, including luxury and standard makes such as Benz, BMW, Cadillac, Ferrari, Ford, Lamborghini, Porsche, Rolls-Royce, and Toyota.

## 🛠️ Methodology

### Data Preprocessing
To prepare raw images for training, the following preprocessing steps were implemented using TensorFlow/Keras `ImageDataGenerator`:
* **Resizing:**
    * $224 \times 224$ pixels for ResNet50.
    * $299 \times 299$ pixels for InceptionV3.
* **Normalization:** Pixel scaling using the respective `preprocess_input` functions for each architecture.
* **Augmentation:** Real-time augmentation including Shear Range (0.2), Zoom Range (0.2), and Horizontal Flips to prevent overfitting.

### Model Architectures

#### 1. ResNet50
* **Backbone:** Pre-trained on ImageNet, frozen to act as a feature extractor.
* **Head:** Global Average Pooling, followed by a Dense layer (512 units, ReLU), Dropout (0.5), and a Softmax output layer.
* **Optimizer:** Adam.

#### 2. InceptionV3
* **Backbone:** Pre-trained on ImageNet, utilizing factorized convolutions and inception modules.
* **Head:** Flatten/Global Pooling, followed by a Dense layer (1024 units, ReLU), Dropout (0.5), and a Softmax output layer.
* **Optimizer:** RMSprop.

## 📈 Results & Evaluation

The models were evaluated based on Accuracy, Precision, Recall, and F1-Score.

| Metric | ResNet50 | InceptionV3 |
| :--- | :--- | :--- |
| **Validation Accuracy** | 68.73% | 86.20% |
| **Training Accuracy** | 71.81% | 94.37% |
| **Weighted F1-Score** | 0.68 | 0.87 |
| **Training Loss** | 0.91 | 0.21 |


### Key Observations
* **InceptionV3** demonstrated superior performance, utilizing its wider architecture to better capture fine-grained details. It achieved high precision for distinct luxury shapes like **Bentley** (0.98 Precision).
* **ResNet50** served as a baseline but struggled with visually similar classes, particularly **Toyota** and **Benz**, often confusing them with other standard sedans. However, it performed well on classes with distinct shapes like **Cadillac** (0.92 Precision).

##  Getting Started

### Prerequisites
* Python 3.x
* TensorFlow / Keras
* NumPy
* Matplotlib
* Pandas

### Installation
1.  Clone the repository:
2.  Install dependencies:
    ```bash
    pip install tensorflow numpy matplotlib pandas
    ```

### Usage
1.  **Dataset Setup:** Download the dataset from Kaggle and place it in the project directory. Update the path in the script accordingly.
2.  **Training:** Run the training scripts for the respective models.
3.  **Evaluation:** Use the evaluation scripts to generate confusion matrices and classification reports.



