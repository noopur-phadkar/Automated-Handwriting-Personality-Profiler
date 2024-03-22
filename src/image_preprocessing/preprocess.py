import cv2
import numpy as np
import os

def denoise_image(image):
    # Using Gaussian blur for denoising
    return cv2.GaussianBlur(image, (5, 5), 0)

def binarize_image(image):
    # Binarization using Otsu's thresholding
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def remove_slant(image):
    # Slant correction using Shear Transformation
    (h, w) = image.shape
    shear_factor = 0.2 # This value may need to be adjusted for your dataset
    M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
    shifted = cv2.warpAffine(image, M, (w, h))
    return shifted

def normalize_image(image):
    # Normalizing the image to a standard size
    normalized_image = cv2.resize(image, (256, 256))
    return normalized_image

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = denoise_image(image)
    image = binarize_image(image)
    image = remove_slant(image)
    processed_image = normalize_image(image)
    return processed_image

# Example usage for a directory
def preprocess_dataset(directory):
    preprocessed_images = []
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        preprocessed_image = preprocess_image(image_path)
        preprocessed_images.append(preprocessed_image)
    return preprocessed_images
