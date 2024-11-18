# pylint: disable=no-member
"""
Author  : Vian Sebastian B
Version : 1
Date    : 17/11/2024

"circles_modules.py "
This module contains circle-specific handling functions

Key Components:
- Blob detection and handling
- Scoring computation

Usage:
- Serves as circle-specific handling module
"""

import numpy as np
import cv2
from core_modules.base_preprocessing_modules import soft_morph_open


def draw_full_contours(contours, cont_image, radius=7):
    """
    Draws filled circles at the center of each contour on the given image.

    Parameters:
      - contours (list): List of contours to be drawn.
      - cont_image (numpy.ndarray): Image on which contours will be drawn.
      - radius (int): Radius of the filled circles to be drawn at the contour centers.

    Returns:
      - numpy.ndarray: Image with contours drawn as filled circles.
    """
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # draw filled circle at contour center
            cv2.circle(cont_image, (cX, cY), radius, (0, 255, 0), -1)

    return cont_image


def extract_and_draw_contours(image):
    """
    Extracts all contours from an image and draws filled circles at each contour center.

    Parameters:
      - image (numpy.ndarray): Input binary or grayscale image for contour extraction.

    Returns:
        tuple: (contours, contour_image)
            - contours (list): List of contours detected in the image.
            - contour_image (numpy.ndarray): Image with drawn contours as filled circles.
    """
    contours, _ = cv2.findContours(
        image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    unique_values = []
    for columns in image:
        for pixel in columns:
            if pixel not in unique_values:
                unique_values.append(pixel)

    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contour_image = draw_full_contours(contours, contour_image)

    return contours, contour_image


def extract_and_draw_circle_contours(image):
    """
    Extracts contours from an image and filters them to find circular contours.
    Draws filled circles at the center of each circular contour.

    Parameters:
      - image (numpy.ndarray): Input binary or grayscale image for circular contour extraction.

    Returns:
        tuple: (circle_contours, contour_image)
            - circle_contours (list): List of contours approximated as circles.
            - contour_image (numpy.ndarray): Image with drawn circular contours as filled circles.
    """
    contours, _ = cv2.findContours(
        image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    circle_contours = []
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        # approximate the enclosing circle for each contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)

        # compute actual contour area
        contour_area = cv2.contourArea(contour)

        # 'circular' tolerance
        if radius < 5:
            if 0.6 <= contour_area / circle_area <= 1.4:
                circle_contours.append(contour)
        else:
            if 0.8 <= contour_area / circle_area <= 1.2:
                circle_contours.append(contour)

    contour_image = draw_full_contours(circle_contours, contour_image)

    return circle_contours, contour_image


def final_scoring(new_student, processed_student, master_contours):
    """
    Evaluates student answers by comparing with master contours, and scores based on detected differences.

    Parameters:
      - new_student (numpy.ndarray): Student's answer image.
      - processed_student (numpy.ndarray): Preprocessed student's answer image.
      - master_contours (list): Contours from the master answer key image.

    Returns:
        tuple: (stu_final_score, student_correction, detected_total_questions, detected_mistakes)
            - stu_final_score (float): The final score for the student.
            - student_correction (list): List of corrections for student answers.
            - detected_total_questions (int): Total number of detected questions.
            - detected_mistakes (int): Number of detected mistakes in the student's answers.
    """
    # processing mistake location and count
    test_answer = processed_student.copy()
    check_answers = draw_full_contours(master_contours, test_answer)
    final_sheet = soft_morph_open(check_answers)
    final_contours, _ = extract_and_draw_circle_contours(final_sheet)

    # mistakes, etc. computation
    mistakes = len(final_contours)
    total_questions = len(master_contours)
    print(f'total_questions: {total_questions}, mistakes: {mistakes}')
    final_score = ((total_questions - mistakes) / total_questions) * 100
    print(f'final score: {final_score}')

    # retrieve student correction
    student_correction = cv2.cvtColor(new_student, cv2.COLOR_GRAY2BGR)
    student_correction = draw_full_contours(
        master_contours, student_correction)

    return final_score, student_correction, total_questions, mistakes
