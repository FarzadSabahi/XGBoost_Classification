# XGBoost Image Classification on CIFAR-10

This project demonstrates how to use the XGBoost machine learning library for image classification on the CIFAR-10 dataset. While XGBoost is typically used for structured/tabular data, this serves as an educational exercise to show its flexibility. It should be noted that convolutional neural networks (CNNs) are generally more appropriate for image classification tasks due to their ability to preserve and utilize the spatial information in images.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have met the following requirements:
- You have installed Python 3.x.
- You have a xboost is installed.
  
      pip install xgboost

### Project Explanation
- **Loading Data:** The CIFAR-10 dataset is loaded directly from the Keras datasets module for convenience.
- **Preprocessing:** Images are flattened into vectors because XGBoost cannot handle image data in its original 3D format. Each 32x32 color image is turned into a single 3072-length vector (32 height x 32 width x 3 color channels).
- **Model Training:** An XGBoost classifier is created with basic parameters, and the model is trained on the preprocessed images.
- **Evaluation:** The model's accuracy is evaluated using the test set.

It's important to note that the model's performance is typically lower compared to a deep learning approach. This project is primarily for educational purposes to demonstrate XGBoost's application outside of its usual domain.
