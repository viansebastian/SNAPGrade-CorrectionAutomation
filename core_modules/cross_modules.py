# pylint: disable=no-member
"""
Author  : Vian Sebastian B
Version : 2
Date    : 20/11/2024

"cross_modules.py "
This module contains cross-specific handling functions

Key Components:
- YOLO for cross detection
- Box contour handling
- Cross scoring computation

Usage:
- Serves as cross-specific handling module

V1 - V2: utilized avg for box tagging instead of max
        to avoid overlapping boxes
"""

import numpy as np
import cv2
import torch
from core_modules.base_preprocessing_modules import soft_morph_open

# print(cv2.__version__)
# print(np.__version__)


def yolo_catch_image(model, image):
    """
    Uses a YOLO model to detect objects in an image and returns predictions and bounding box coordinates.

    Parameters:
        - model: YOLO model for object detection.
        - image (numpy.ndarray): Input image in grayscale format.

    Returns:
        tuple: (pred, boxes, coords)
            - pred: Model prediction.
            - boxes: Detected bounding boxes.
            - coords: Bounding box coordinates in (center x, center y, width, height).
    """
    input = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    res = model.predict(input)
    pred = res[0]
    boxes = pred.boxes
    coords = boxes.xywh
    # confidences = boxes.conf
    # class_ids = boxes.cls

    return pred, boxes, coords


def get_max_width_height(master_coords, student_coords):
    """
    Computes the maximum width and height from bounding box coordinates of master and student images.

    Parameters:
        - master_coords (torch.Tensor or numpy.ndarray): Master bounding box coordinates.
        - student_coords (torch.Tensor or numpy.ndarray): Student bounding box coordinates.

    Returns:
        - int: Average of the maximum width and maximum height across both sets of coordinates.
    """
    max_width = 0
    max_height = 0

    # combine both sets of coordinates
    all_coords = torch.cat([master_coords, student_coords], dim=0) if isinstance(
        master_coords, torch.Tensor) else np.vstack((master_coords, student_coords))

    # traverse all bounding box dimensions
    for coord in all_coords:
        # convert tensor to NumPy array if needed
        if isinstance(coord, torch.Tensor):
            coord = coord.cpu().numpy()
        _, _, width, height = coord.astype(int)

        # update maximum width and height
        max_width = max(max_width, width)
        max_height = max(max_height, height)

    # return max(max_width, max_height)
    return int((max_width + max_height) / 2)

def get_avg_width_height(master_coords, student_coords):
    """
    Computes the average width and height from master and student bounding box coordinates.

    Args:
        master_coords (torch.Tensor or numpy.ndarray): Bounding box coordinates for the master image (center_x, center_y, width, height).
        student_coords (torch.Tensor or numpy.ndarray): Bounding box coordinates for the student image (center_x, center_y, width, height).

    Returns:
        float: Average of width and height across both sets of coordinates.
    """
    # Combine both sets of coordinates
    all_coords = (
        torch.cat([master_coords, student_coords], dim=0)
        if isinstance(master_coords, torch.Tensor)
        else np.vstack((master_coords, student_coords))
    )

    # Initialize variables for summing widths and heights
    total_width = 0
    total_height = 0
    num_coords = all_coords.shape[0]

    # Traverse all bounding box dimensions
    for coord in all_coords:
        # Convert tensor to NumPy array if needed
        if isinstance(coord, torch.Tensor):
            coord = coord.cpu().numpy()
        _, _, width, height = coord.astype(int)

        # Sum up width and height
        total_width += width
        total_height += height

    # Compute average width and height
    avg_width = total_width / num_coords
    avg_height = total_height / num_coords

    # Return the average of width and height
    return round((avg_width + avg_height) / 2)

def mark_ans_box(
    image,
    coords,
    shape="box",
    size=10,
    color=(
        0,
        255,
        0),
        thickness=-
        1):
    """
    Draw a fixed-size filled box or circle at the center of each detected box.

    Parameters:
        - image (numpy.ndarray): The input image where the shapes will be drawn.
        - boxes (object): The detected boxes (used for validation, if needed).
        - coords (torch.Tensor or numpy.ndarray): Box coordinates (center x, center y, width, height).
        - shape (str): The shape to draw ("box" or "circle"). Default is "box".
        - size (int): The size (side length or radius) of the shape. Default is 10.
        - color (tuple): The color of the shape in (B, G, R). Default is green.
        - thickness (int): Thickness of the shape. Use -1 for filled shapes. Default is -1.

    Returns:
        - numpy.ndarray: The image with the shapes drawn on it.
    """
    input_img = image.copy()
    for coord in coords:
        # Convert tensor to NumPy array if needed and cast to integers
        if isinstance(coord, torch.Tensor):
            coord = coord.cpu().numpy()
        center_x, center_y, _, _ = coord.astype(int)

        if shape == "box":
            # Calculate the top-left and bottom-right coordinates of the box
            top_left = (center_x - size // 2, center_y - size // 2)
            bottom_right = (center_x + size // 2, center_y + size // 2)
            # Draw the box
            cv2.rectangle(input_img, top_left, bottom_right, color, thickness)
        elif shape == "circle":
            # Draw the circle
            cv2.circle(input_img, (center_x, center_y), size, color, thickness)
        else:
            raise ValueError("Shape must be either 'box' or 'circle'.")

    return input_img


def draw_filled_boxes(contours, cont_image):
    """
    Draws filled boxes inside each white region defined by contours in a binary image.

    Parameters:
        - contours (list): List of contours.
        - cont_image (numpy.ndarray): Grayscale image containing contours.

    Returns:
        - numpy.ndarray: Image with filled boxes drawn within contours.
    """
    input_img = cont_image.copy()
    input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
    for contour in contours:
        # Get the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the center of the rectangle
        center_x, center_y = x + w // 2, y + h // 2

        # Define the size of the filled box
        size = min(w, h) // 2  # Adjust box size to fit within the region

        # Draw a filled rectangle centered within the bounding box
        top_left = (center_x - size, center_y - size)
        bottom_right = (center_x + size, center_y + size)
        cv2.rectangle(input_img, top_left, bottom_right, (0, 255, 0), -1)

    return input_img


def draw_filled_boxes_fetch_contours(image):
    """
    Detects contours in an image and draws filled boxes within each white region.

    Parameters:
        - image (numpy.ndarray): Grayscale image.

    Returns:
        tuple: (contours, result_image)
            - contours: List of detected contours.
            - result_image: Image with filled boxes drawn within contours.
    """
    input_img = image.copy()

    contours, _ = cv2.findContours(
        input_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_image = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x, center_y = x + w // 2, y + h // 2
        size = min(w, h) // 2
        top_left = (center_x - size, center_y - size)
        bottom_right = (center_x + size, center_y + size)
        cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), -1)

    return contours, result_image


def draw_box_fill_space_overlap(
        master_contours,
        student_contours,
        image_shape,
        overlap_threshold=0.1):
    """
    Draws contours with conditions:
    - Regions with >=10% overlap between student and master contours are filled green.
    - Non-overlapping or <10% overlap contours from student_contours are left white.

    Parameters:
        - master_contours (list): Contours from the master (reference) image.
        - student_contours (list): Contours from the student (target) image.
        - image_shape (tuple): Shape of the new image (height, width, channels).
        - overlap_threshold (float): Minimum overlap ratio to classify as overlapping (default: 10%).

    Returns:
        - numpy.ndarray: Image with processed contours.
    """
    # Create a blank image for the result
    result_img = np.zeros(image_shape, dtype=np.uint8)

    # Create masks for master and student contours
    master_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    # student_mask = np.zeros(image_shape[:2], dtype=np.uint8)

    # Draw the master contours on the master mask
    cv2.drawContours(master_mask, master_contours, -
                     1, 255, thickness=cv2.FILLED)

    for contour in student_contours:
        # Create an individual mask for the current student contour
        single_student_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.drawContours(single_student_mask, [
                         contour], -1, 255, thickness=cv2.FILLED)

        # Find overlapping regions using bitwise AND
        overlap_mask = cv2.bitwise_and(master_mask, single_student_mask)

        # Calculate the overlap ratio
        overlap_area = np.sum(overlap_mask > 0)
        student_area = np.sum(single_student_mask > 0)
        overlap_ratio = overlap_area / student_area if student_area > 0 else 0

        # Decide color based on overlap percentage
        if overlap_ratio >= overlap_threshold:
            # Fill green for significant overlap
            result_img[overlap_mask > 0] = (0, 255, 0)
        else:
            # Fill white for non-overlapping or low-overlap regions
            result_img[single_student_mask > 0] = (255, 255, 255)

    return result_img


def get_cross_answers(model, master_image, student_image):
    """
    Detects bounding boxes on master and student images, draws answer boxes, and returns processed images.

    Parameters:
        - model: YOLO model for object detection.
        - master_image (numpy.ndarray): Master answer image.
        - student_image (numpy.ndarray): Student answer image.

    Returns:
        tuple: (master_img, student_img)
            - master_img: Master image with drawn answer boxes.
            - student_img: Student image with drawn answer boxes.
    """
    _, _, master_coords = yolo_catch_image(
        model, master_image)
    _, _, student_coords = yolo_catch_image(
        model, student_image)

    box_size = get_avg_width_height(master_coords, student_coords)

    master_img = mark_ans_box(
        master_image,
        master_coords,
        shape='box',
        size=box_size,
        color=(
            0,
            255,
            0),
        thickness=-1)
    student_img = mark_ans_box(
        student_image,
        student_coords,
        shape='box',
        size=box_size,
        color=(
            0,
            255,
            0),
        thickness=-1)

    return master_img, student_img


def box_contour_handling(master_image, student_image):
    """
    Handles contours by comparing master and student images, filling overlapping regions.

    Parameters:
        - master_image (numpy.ndarray): Master answer image.
        - student_image (numpy.ndarray): Student answer image.

    Returns:
        tuple: (master_contours, student_contours, stu_mistake_loc_final)
            - master_contours: Contours detected in master image.
            - student_contours: Contours detected in student image.
            - stu_mistake_loc_final: Processed image with overlapping areas marked.
    """
    master_contours, _ = draw_filled_boxes_fetch_contours(
        master_image)
    student_contours, _ = draw_filled_boxes_fetch_contours(
        student_image)

    shape = (student_image.shape[0], student_image.shape[1], 3)
    stu_mistake_loc_cont_img = draw_box_fill_space_overlap(
        master_contours, student_contours, shape)
    stu_mistake_loc_cont_img = cv2.cvtColor(
        stu_mistake_loc_cont_img, cv2.COLOR_BGR2GRAY)
    _, result_img = cv2.threshold(
        stu_mistake_loc_cont_img, 254, 255, cv2.THRESH_BINARY)
    stu_mistake_loc_final = soft_morph_open(result_img)

    return master_contours, student_contours, stu_mistake_loc_final


def final_scoring_cross(student_img, master_contours, student_mistake_loc):
    """
    Evaluates student answers by comparing with master contours, and scores based on detected differences.

    Parameters:
        - new_student (numpy.ndarray): Student's answer image.
        - master_contours (list): Contours from the master answer key image.
        - student_mistake_loc(numpy.ndarray): Student's mistake location.

    Returns:
        tuple: (stu_final_score, student_correction, detected_total_questions, detected_mistakes)
            - stu_final_score (float): The final score for the student.
            - student_correction (list): List of corrections for student answers.
            - detected_total_questions (int): Total number of detected questions.
            - detected_mistakes (int): Number of detected mistakes in the student's answers.
    """
    mistake_contours, _ = draw_filled_boxes_fetch_contours(student_mistake_loc)

    student_correction = student_img.copy()
    student_correction = draw_filled_boxes(master_contours, student_correction)

    mistakes = len(mistake_contours)
    total_questions = len(master_contours)
    final_score = ((total_questions - mistakes) / total_questions) * 100
    final_score = round(final_score, 2)
    print(f'total_questions: {total_questions}, mistakes: {mistakes}')
    print(f'final score: {final_score}')

    return final_score, student_correction, total_questions, mistakes
