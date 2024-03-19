import numpy as np
from skimage.feature import hog
from preprocess import preprocess_image

def extract_features(image):
    # Here we use HoG features as an example, but you should
    # explore and extract features that are relevant for handwriting analysis
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(3, 3), visualize=True, multichannel=False)
    return features

# Example usage
# image_features = extract_features(preprocessed_image)
