import cv2
import numpy as np


def vertical_projection_profile(image):
    # Summing up the pixels in vertical direction
    v_projection = np.sum(image, axis=0)
    return v_projection


def segment_on_vpp(image, v_projection):
    # Finding where the columns of white start and end
    start = None
    end = None
    segments = []
    for i, val in enumerate(v_projection):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            end = i
            segments.append((start, end))
            start = None
    # Don't forget the last segment
    if start is not None:
        segments.append((start, len(v_projection)))
    return segments


def extract_lines(image):
    # Assuming that the image is binary inverted (text is white on black background)
    # Summing up the pixels in horizontal direction to find lines
    horizontal_projection = np.sum(image, axis=1)
    line_start = None
    line_end = None
    lines = []
    for i, val in enumerate(horizontal_projection):
        if val > 0 and line_start is None:
            line_start = i
        elif val == 0 and line_start is not None:
            line_end = i
            lines.append(image[line_start:line_end, :])
            line_start = None
    # Don't forget the last line
    if line_start is not None:
        lines.append(image[line_start:, :])
    return lines


def segment_text(image):
    # First, let's segment the image into lines
    lines = extract_lines(image)
    words = []
    for line in lines:
        # For each line, we apply vertical projection profile
        vpp = vertical_projection_profile(line)
        # Get the word segments
        word_segments = segment_on_vpp(line, vpp)
        # Now we have the start and end points for each word in the line
        for start, end in word_segments:
            word = line[:, start:end]
            words.append(word)
    return words


# Example usage
# Load the image and invert it for processing
image = cv2.imread('path_to_preprocessed_image', cv2.IMREAD_GRAYSCALE)
binary_image = cv2.bitwise_not(image)
words = segment_text(binary_image)
# Now 'words' contains segmented words from the handwritten text
