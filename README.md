# Automated Handwriting Personality Profiler

## Project Description
The Automated Handwriting Personality Profiler aims to predict the Big Five personality traits from images of handwritten text. Utilizing advancements in machine learning and computer vision, this project analyzes handwriting characteristics and infers personality traits. This interdisciplinary venture combines psychology and AI, leveraging the hypothesis that handwriting styles can reflect distinct personality traits.

## Technologies Used
- **Python**: The core programming language used for developing the project, chosen for its extensive libraries and community support.
- **Machine Learning and Computer Vision Libraries**: Including TensorFlow for building the CNN model, OpenCV for image processing, and NumPy for numerical computing.

### System Architecture
The system architecture involves several key components working together to analyze handwriting and generate personality profiles:

1. **Image Preprocessing**: Processes raw handwritten text data to ensure data quality and consistency.
2. **Feature Extraction**: Involves various techniques such as image pre-processing, segmentation, and CNN-based feature extraction to capture handwriting nuances.
3. **Text Segmentation Pipeline**: Divides handwritten text into meaningful segments using edge detection algorithms, Vertical Projection Profiles, and adaptive segmentation techniques.
4. **CNN Model**: Interprets the extracted features and predicts personality traits based on handwriting characteristics.

![System Architecture](https://github.com/noopur-phadkar/Automated-Handwriting-Personality-Profiler/assets/98292727/4a54fc10-6329-42d4-a5f2-593119ef1dfb)

## Key Components
- **Data Preprocessing**: Handled by Python scripts, preparing handwritten text data for feature extraction.
- **Feature Extraction**: Uses advanced techniques for extracting features from handwritten text.
- **Text Segmentation Pipeline**: Implements a sophisticated pipeline for text segmentation.
- **CNN Model**: A Convolutional Neural Network trained to predict personality traits.
- **Azure Integration**: Utilizes Azure Blob Storage, Machine Learning services, and DevOps for efficient data management and processing.

## Data Structure
The dataset consists of images of handwritten text, organized as follows:
```plaintext
dataset/
├── training_set/
│   ├── Agreeableness/
│   ├── Conscientiousness/
│   ├── Extraversion/
│   ├── Neuroticism/
│   └── Openness/
└── test_set/
    ├── Agreeableness/
    ├── Conscientiousness/
    ├── Extraversion/
    ├── Neuroticism/
    └── Openness/
```
Each subfolder contains images used for training and evaluating the model.

## Usage
To use the project, execute the following steps:
1. Preprocess the dataset with `preprocess.py`.
2. Extract features using `feature_extraction.py`.
3. Segment text with `segmentation.py`.
4. Define the CNN model in `model.py`.
5. Train and evaluate the model using `train.py`.

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### Notes:
- Update the image link in the README to the actual path of your system architecture diagram, if available.
- The sections commented out (like detailed instructions, Azure integration) can be filled in or adjusted based on your actual project setup and deployment strategies.
- Make sure to add any additional instructions or notes that might be helpful for users or contributors to understand and run the project effectively.

This README provides a comprehensive overview of your project, focusing on its key components, usage, and data structure.