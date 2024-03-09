# Automated Handwriting Personality Profiler

## Project Objectives
The Automated Handwriting Personality Profiler aims to harness the power of machine learning and computer vision to analyze handwritten texts and deduce nuanced characteristics indicative of the writer's personality traits. This project highlights the interdisciplinary nature of artificial intelligence, combining aspects of psychology, computer science, and data analysis to create a tool that can offer insightful personality assessments based on handwriting analysis. The objective is to demonstrate how automated systems can perform complex interpretations traditionally done by human experts, providing a scalable and innovative approach to personality profiling.

## Key Components
The project is structured around several key components, each crucial for processing handwritten text and generating personality profiles:
- **Data Preprocessing**: Python scripts handle data preprocessing tasks, ensuring that handwritten text data is cleaned, formatted, and prepared for feature extraction.
- **Feature Extraction**: Advanced techniques are employed to extract features from handwritten text, including image preprocessing, segmentation, and Convolutional Neural Networks (CNN).
- **Text Segmentation Pipeline**: The project implements a sophisticated text segmentation pipeline. It utilizes edge detection algorithms for line separation, Vertical Projection Profiles for word boundaries, and adaptive Connected Component Analysis (CCA) or Stroke Width Transform (SWT) for letter segmentation.
- **CNN Model**: A Convolutional Neural Network trained to interpret these features and predict personality traits based on handwriting characteristics.
- **Azure Integration**: Azure Blob Storage is utilized to store handwritten text images, facilitating easy access and management. Azure Machine Learning services are leveraged to process images and extract features, utilizing cloud computing power for efficient analysis. Azure DevOps is employed to host the project, manage development pipelines, and ensure smooth collaboration among team members.

## Technologies Used
- **Python**: The core programming language used for developing the project, chosen for its extensive libraries and community support.
- **Machine Learning and Computer Vision Libraries**: Including TensorFlow for building the CNN model, OpenCV for image processing, and NumPy for numerical computing.
- **Azure Services**: Azure Blob Storage for storing handwriting images, Azure Machine Learning services for model training and deployment, and Azure DevOps for continuous integration and delivery.

### System Architecture
The system architecture of the Automated Handwriting Personality Profiler involves several components working together seamlessly to achieve the project's objectives:

<p align="center">
  <img width="600" alt="System Architecture" src="https://github.com/noopur-phadkar/Automated-Handwriting-Personality-Profiler/assets/98292727/4a54fc10-6329-42d4-a5f2-593119ef1dfb">
</p>

1. **Image Preprocessing**: Raw handwritten text data is fed into the system, where Python scripts handle preprocessing tasks to ensure data quality and consistency.

2. **Feature Extraction**: Extracting features from handwritten text involves various techniques such as image pre-processing, segmentation, and CNN-based feature extraction. These features are essential for capturing the nuances and characteristics of the handwriting.

3. **Text Segmentation Pipeline**: The text segmentation pipeline is a crucial component of the system, responsible for dividing handwritten text into meaningful segments such as lines, words, and letters. It utilizes edge detection algorithms, Vertical Projection Profiles, and adaptive segmentation techniques for accurate segmentation.

4. **Azure Integration**: Azure services play a vital role in the project's architecture. Azure Blob Storage is used for storing handwritten text images securely and efficiently. Azure Machine Learning services leverage cloud computing power to process images and extract features, enabling scalable and efficient analysis. Azure DevOps provides a robust platform for hosting the project, managing development pipelines, and facilitating collaboration among team members.

### Usage
1. Data Preprocessing: Use Python scripts provided in the repository to preprocess handwritten text data.
2. Feature Extraction: Implement image pre-processing techniques, segmentation, and CNN-based feature extraction using provided scripts or custom implementations.
3. Text Segmentation: Utilize the text segmentation pipeline with edge detection, Vertical Projection Profiles, and adaptive CCA or SWT for letter segmentation.
4. Azure Integration: Set up Azure Blob Storage for image storage, Azure Machine Learning services for image processing and feature extraction, and Azure DevOps for project hosting and pipeline management.

### License
This project is licensed under the MIT License. See the `LICENSE` file for details.

<!--
## Detailed Instructions
To set up and run the project locally, follow these steps:
1. **Environment Setup**: Clone the repository and create a virtual environment. Install required dependencies using `pip install -r requirements.txt`.
2. **Data Preparation**: Upload your dataset of handwritten images to Azure Blob Storage. Ensure images are pre-labeled with personality traits for training the CNN.
3. **Running the Pre-processing Script**: Execute the pre-processing script to normalize and segment the images. `python preprocess.py`.
4. **Feature Extraction**: Run the feature extraction script to analyze handwriting characteristics from the processed images. `python feature_extraction.py`.
5. **Model Training**: Use the extracted features to train the CNN. Adjust parameters as needed to improve accuracy. `python train_model.py`.
6. **Evaluation and Prediction**: Evaluate the model's performance with a test set and use the model to predict personality traits from new handwritten samples. `python predict.py`.
-->
