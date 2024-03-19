import cv2
import numpy as np
import os


def denoise_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def remove_slant(image):  # TODO
    # This is a placeholder for slant correction code.
    # You would need to implement or find a suitable algorithm to correct the slant.
    return image


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = denoise_image(image)
    # Binarization
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Remove slant from the image
    processed_image = remove_slant(binary_image)
    # Resize to fixed size for CNN input
    processed_image = cv2.resize(processed_image, (256, 256))
    return processed_image


def preprocess_dataset(directory):
    preprocessed_images = []
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        preprocessed_image = preprocess_image(image_path)
        preprocessed_images.append(preprocessed_image)
    return preprocessed_images


if __name__ == '__main__':
    # Example usage
    preprocessed_images = preprocess_dataset('path_to_your_dataset')
