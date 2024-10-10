# Rock-Paper-Scissors Classification

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)

## Introduction

Welcome to the **Rock-Paper-Scissors Classification** project! This project leverages deep learning techniques to build a Convolutional Neural Network (CNN) that can accurately classify images of rock, paper, and scissors. Whether you're interested in computer vision, game AI, or machine learning, this project provides a comprehensive example of building and deploying an image classification model using TensorFlow and Keras in Google Colab.

## Features

- **Data Augmentation:** Enhances the dataset with transformations like rotation, shifting, shearing, and flipping to improve model robustness.
- **Custom CNN Architecture:** Utilizes multiple convolutional and pooling layers to effectively extract features from images.
- **Real-time Classification:** Allows users to upload images and get instant predictions.
- **Visualization:** Provides plots for training and validation accuracy and loss to monitor model performance.
- **Early Stopping:** Implements a callback to halt training once the desired accuracy is achieved, optimizing training time.

## Dataset

The dataset used in this project consists of images representing the three classes: **Rock**, **Paper**, and **Scissors**. The dataset is sourced from [Dicoding Academy](https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip) and includes a variety of images to ensure the model can generalize well.

**Dataset Structure:**
rockpaperscissors/ ├── rps-cv-images/ │ ├── rock/ │ ├── paper/ │ └── scissors/

## Model Architecture

The CNN model is built using TensorFlow's Keras API with the following architecture:

- **Convolutional Layers:** 4 layers with increasing filters (32, 64, 128, 128) to capture spatial hierarchies.
- **MaxPooling Layers:** Reduce spatial dimensions and control overfitting.
- **Flatten Layer:** Converts 2D feature maps to 1D feature vectors.
- **Dense Layers:** Includes a fully connected layer with 512 units and a softmax output layer for classification into three categories.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
```

## Getting Started
### Prerequisites
- Google Account: Required to use Google Colab.
- Internet Connection: To download the dataset and run the notebook in Colab.

### Setup
1. Clone the Repository:
```bash
git clone https://github.com/brilianap/rock-paper-scissors-classification.git
cd rock-paper-scissors-classification
```
2. Access the Google Colab Notebook:
- Open the Rock-Paper-Scissors Classification Notebook directly in Google Colab.
- Alternatively, navigate to the repository on GitHub and click on the notebook file to open it in Colab.
