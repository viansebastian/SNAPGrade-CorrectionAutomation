# pylint: disable=no-member
"""
Author  : Vian Sebastian B
Version : 1
Date    : 17/11/2024

"base_preprocessing_modules.py "
This module contains base preprocessing modules, used for both
circles and cross tasks.

Key Components:
- Resizing image
- Logarithmic transformation
- Contrast stretching
- Gaussian blurring
- Adaptive Gaussian blurring
- CLAHE equalization
- Otsu thresholding
- Canny Edge detection
- Automatic Perspective Warping
- Resizing uniformization
- Image opening

Usage:
- Serves as the main preprocessing module
"""

import cv2
import numpy as np

# print(cv2.__version__)
# print(np.__version__)


def resize_image(image, target_size=(800, 1000)):
    """
    Resize the image while maintaining its aspect ratio.

    Parameters:
      - image (ndarray): Input image to be resized.
      - target_size (tuple): Target size for the image as (height, width).

    Returns:
      - resized_image (ndarray): Resized image with maintained aspect ratio.
    """
    h, w = image.shape[:2]
    scale = min(target_size[1] / w, target_size[0] / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image


def logarithmic_transformation(image, epsilon=1e-5):
    """
      Apply logarithmic transformation to enhance the image contrast, with zero values handling.

      Parameters:
        - image (ndarray): Input image.
        - epsilon (float): Small value added to avoid log(0), defaults to 1e-5.

      Returns:
        - log_image (ndarray): Logarithmically transformed image.
    """
    c = 255 / np.log(1 + np.max(image))
    # Epsilon zero-handling technique
    log_image = c * (np.log(1 + image + epsilon))
    log_image = np.array(log_image, dtype=np.uint8)

    return log_image


def contrast_stretching(image):
    """
    Perform contrast stretching to improve contrast.

    Parameters:
      - image (ndarray): Input grayscale image.

    Returns:
      - stretched (ndarray): Contrast-stretched image.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = (image - min_val) * (255 / (max_val - min_val))
    return stretched.astype(np.uint8)


def gaussian_blur(image, mode='Soft'):
    """
    Apply Gaussian blur to the image with varying strength.

    Parameters:
      - image (ndarray): Input image.
      - mode (str): Blurring strength, can be 'Soft', 'Medium', or 'Hard'.

    Returns:
      - blurred_image (ndarray): Blurred image.
    """
    if mode == 'Soft':
        kernel_size = (3, 3)
    elif mode == 'Medium':
        kernel_size = (5, 5)
    elif mode == 'Hard':
        kernel_size = (7, 7)
    else:
        raise ValueError("Mode must be 'Soft', 'Medium', or 'Hard'")

    return cv2.GaussianBlur(image, kernel_size, 0)


def measure_blurriness(image):
    """
    Measure the blurriness of the image using the Laplacian operator.

    Parameters:
      - image (ndarray): Input grayscale image.

    Returns:
      - variance (float): Variance of the Laplacian, representing the image sharpness.
    """
    # Laplacian operator fpr edge detection
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # Laplacian variance
    variance = laplacian.var()

    return variance


def adaptive_gaussian_blur(image, desired_blur=100, max_iterations=100):
    """
    Apply Gaussian blur adaptively until a specified level of blurriness is achieved.

    Parameters:
      - image (ndarray): Input grayscale image.
      - desired_blur (float): Target blurriness level.
      - max_iterations (int): Maximum iterations to achieve the desired blur.

    Returns:
      - final_blurred_img (ndarray): Blurred image meeting the desired blur criteria.
    """
    # initial blur level
    initial_blur = measure_blurriness(image)

    # start kernel size
    kernel_size = 5

    for iteration in range(max_iterations):
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        current_blur = measure_blurriness(blurred_image)

        # stop if current blur exceeds desired blur
        if current_blur > desired_blur:
            kernel_size += 2
        else:
            break

    final_blurred_img = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    final_blur = measure_blurriness(final_blurred_img)

    print(
        f"Initial Blur: {initial_blur}, Final Blur: {final_blur}, Kernel Size: {kernel_size}, Iterations: {iteration+1}")

    return final_blurred_img


def clahe_equalization(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the image.

    Parameters:
      - image (ndarray): Input grayscale image.

    Returns:
      - equalized_img (ndarray): CLAHE-equalized image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_img = clahe.apply(image)
    return equalized_img


def otsu_thresholding(image):
    """
    Apply Otsu's thresholding method to binarize the image.

    Parameters:
      - image (ndarray): Input grayscale image.

    Returns:
      - binary_image (ndarray): Binarized image using Otsu's thresholding.
    """
    _, binary_image = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_image


def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Detect edges in the image using the Canny edge detection algorithm.

    Parameters:
      - image (ndarray): Input grayscale image.
      - low_threshold (int): Low threshold for Canny edge detection.
      - high_threshold (int): High threshold for Canny edge detection.

    Returns:
      - edges (ndarray): Edge-detected image.
    """
    return cv2.Canny(image, low_threshold, high_threshold)


def find_extreme_corners(contours):
    """
    Find the extreme corners (top-left, top-right, bottom-left, bottom-right) of the image based on contours.

    Parameters:
      - contours (list): List of contours detected in the image.

    Returns:
      - corners (tuple): Four corner points (top-left, top-right, bottom-left, bottom-right) as tuples.
    """
    all_points = np.vstack(contours)
    top_left = all_points[np.argmin(all_points[:, :, 0] + all_points[:, :, 1])]
    bottom_right = all_points[np.argmax(
        all_points[:, :, 0] + all_points[:, :, 1])]
    top_right = all_points[np.argmax(
        all_points[:, :, 0] - all_points[:, :, 1])]
    bottom_left = all_points[np.argmin(
        all_points[:, :, 0] - all_points[:, :, 1])]
    return top_left[0], top_right[0], bottom_left[0], bottom_right[0]


def apply_perspective_transformation(image, corners):
    """
    Apply perspective transformation to the image based on the provided corner points.

    Parameters:
      - image (ndarray): Input image.
      - corners (tuple): Coordinates of the four corners to transform.

    Returns:
      - warped (ndarray): Warped image after perspective transformation.
    """
    tl, tr, bl, br = corners
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]
    ], dtype="float32")

    src_pts = np.array([tl, tr, bl, br], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


def automatic_warp_transformation(image, target_size=(800, 1000)):
    """
    Perform automatic warp transformation for document alignment and resizing.

    Parameters:
      - image (ndarray): Input image.
      - target_size (tuple): Target size as (height, width) for resizing.

    Returns:
      - warped_image (ndarray): Warped and resized image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = resize_image(gray_image, target_size)
    brightened_image = logarithmic_transformation(resized_image)
    contrast_image = contrast_stretching(brightened_image)
    blurred_image = gaussian_blur(contrast_image, mode='Soft')
    binary_image = otsu_thresholding(blurred_image)
    edges = canny_edge_detection(binary_image)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Getting Contours (Drawing Contours in image, useful for debugging)
    contour_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    corners = find_extreme_corners(contours)
    for corner in corners:
        cv2.circle(contour_image, tuple(corner), 5, (0, 0, 255), -1)

    warped_image = apply_perspective_transformation(resized_image, corners)
    print(f'Initial image {image.shape} processed to {warped_image.shape}')

    return warped_image


def automatic_warp_transformation_v2(image, target_size=(800, 1000)):
    """
    Improved document alignment implementing additional preprocessing.

    Parameters:
      - image (ndarray): Input image.
      - target_size (tuple): Target size as (height, width) for resizing.

    Returns:
      - warped_image (ndarray): Warped and resized image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = resize_image(gray_image, target_size)

    clahe = clahe_equalization(resized_image)
    log_img = logarithmic_transformation(clahe)
    contrast_img = contrast_stretching(log_img)
    blurred_img = gaussian_blur(contrast_img)
    binary_img = otsu_thresholding(blurred_img)
    edges = canny_edge_detection(binary_img)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Getting Contours (Drawing Contours in image, useful for debugging)
    contour_image = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    corners = find_extreme_corners(contours)
    for corner in corners:
        cv2.circle(contour_image, tuple(corner), 5, (0, 0, 255), -1)

    warped_image = apply_perspective_transformation(resized_image, corners)
    print(f'Initial image {image.shape} processed to {warped_image.shape}')

    return warped_image


def image_uniformization(master_image, student_image):
    """
    Resize both master and student images to a uniform size for comparison.

    Parameters:
      - master_image (ndarray): Image of the answer key.
      - student_image (ndarray): Image of the student's answers.

    Returns:
      - resized_master (ndarray): Resized master image.
      - resized_student (ndarray): Resized student image.
    """
    master_shape = master_image.shape
    student_shape = student_image.shape

    master_height = master_shape[0]
    master_width = master_shape[1]

    student_height = student_shape[0]
    student_width = student_shape[1]

    min_height = min(master_height, student_height)
    min_width = min(master_width, student_width)

    resized_master = cv2.resize(master_image, (min_width, min_height))
    resized_student = cv2.resize(student_image, (min_width, min_height))

    print(
        f'master_key {master_image.shape} and student_answer {student_image.shape} uniformed to {resized_master.shape}')

    return resized_master, resized_student


def morph_open(image):
    """
    Apply morphological opening to remove noise from the image.

    Parameters:
      - image (ndarray): Input binary image.

    Returns:
      - dilated_img (ndarray): Image after applying morphological opening.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded_img = cv2.erode(image, kernel, iterations=1)
    dilated_img = cv2.dilate(eroded_img, kernel, iterations=1)

    return dilated_img


def soft_morph_open(image):
    """
    Apply a softer morphological opening operation to the image to remove light noise.

    Parameters:
      - image (ndarray): Input binary image.

    Returns:
      - dilated_img (ndarray): Image after applying soft morphological opening.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_img = cv2.erode(image, kernel, iterations=1)
    dilated_img = cv2.dilate(eroded_img, kernel, iterations=1)

    return dilated_img


def core_preprocessing(image):
    """
    Core preprocessing pipeline for preparing the image.

    Parameters:
      - image (ndarray): Input grayscale image.

    Returns:
      - opened_img (ndarray): Preprocessed binary image.
    """
    blurred_img = gaussian_blur(image, mode='Hard')
    contrast_img = contrast_stretching(blurred_img)
    log_img = logarithmic_transformation(contrast_img)
    binary_img = otsu_thresholding(log_img)
    opened_img = morph_open(binary_img)

    return opened_img


def core_preprocessing_v2(image):
    """
    Improved core processing pipeline, implementing better preprocessing

    Parameters:
      - image (ndarray): Input grayscale image.

    Returns:
      - opened_img (ndarray): Preprocessed binary image.
    """
    clahe_img = clahe_equalization(image)
    blurred_img = adaptive_gaussian_blur(
        clahe_img, desired_blur=100, max_iterations=100)
    contrast_img = contrast_stretching(blurred_img)
    log_img = logarithmic_transformation(contrast_img)
    binary_img = otsu_thresholding(log_img)
    opened_img = morph_open(binary_img)

    return opened_img