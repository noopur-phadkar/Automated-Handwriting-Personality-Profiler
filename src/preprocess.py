import cv2
import numpy as np
import os

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image if necessary
    image = cv2.resize(image, (1280, 720))
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    # Thresholding to get binary image
    _, binary_image = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_image

def preprocess_dataset(directory):
    preprocessed_images = []
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        preprocessed_image = preprocess_image(image_path)
        preprocessed_images.append(preprocessed_image)
    return preprocessed_images

# Example usage
# preprocessed_images = preprocess_dataset('path_to_your_dataset')
