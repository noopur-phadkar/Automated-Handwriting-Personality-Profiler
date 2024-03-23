import os
from src.image_preprocessing.preprocess import preprocess_image
from src.segmentation.segmentation import segment_text
from src.feature_extraction.feature_extraction import extract_features
from src.model.train import train_model
from keras.utils import np_utils
import numpy as np

TRAINING_DATA_PATH = 'dataset/training_set/'
TEST_DATA_PATH = 'dataset/test_set/'

def load_data_from_directory(directory):
    data, labels = [], []
    label_names = os.listdir(directory)
    for label_name in label_names:
        label_dir = os.path.join(directory, label_name)
        if os.path.isdir(label_dir):
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                processed_image = preprocess_image(image_path)
                segments = segment_text(processed_image)
                for segment in segments:
                    features = extract_features(segment)
                    data.append(features)
                    labels.append(label_names.index(label_name))
    data = np.array(data)
    labels = np_utils.to_categorical(labels, num_classes=len(label_names))
    return data, labels

def run_pipeline(training_data_path, test_data_path):
    X_train, y_train = load_data_from_directory(training_data_path)
    X_test, y_test = load_data_from_directory(test_data_path)
    train_model(X_train, y_train, X_test, y_test)
    print("Pipeline execution completed.")

if __name__ == "__main__":
    run_pipeline(TRAINING_DATA_PATH, TEST_DATA_PATH)
