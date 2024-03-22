import cv2
import numpy as np

def vertical_projection_profile(image):
    return np.sum(image, axis=0)

def segment_on_vpp(image, v_projection):
    start, segments = None, []
    for i, val in enumerate(v_projection):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(v_projection)))
    return segments

def extract_lines(image):
    horizontal_projection = np.sum(image, axis=1)
    line_start, lines = None, []
    for i, val in enumerate(horizontal_projection):
        if val > 0 and line_start is None:
            line_start = i
        elif val == 0 and line_start is not None:
            lines.append(image[line_start:i, :])
            line_start = None
    if line_start is not None:
        lines.append(image[line_start:, :])
    return lines

def segment_text(image):
    lines = extract_lines(image)
    words = []
    for line in lines:
        vpp = vertical_projection_profile(line)
        word_segments = segment_on_vpp(line, vpp)
        for start, end in word_segments:
            words.append(line[:, start:end])
    return words
