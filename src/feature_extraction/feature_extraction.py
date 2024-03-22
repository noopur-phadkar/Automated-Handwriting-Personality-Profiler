import numpy as np
from skimage.feature import hog

def extract_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(3, 3), visualize=True, multichannel=False)
    return features
