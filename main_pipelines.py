# pylint: disable=no-member
"""
Author  : Vian Sebastian B
Version : 1
Date    : 17/11/2024

"main_pipelines.py "
This module contains the main pipelines for circle and cross handling

Key Components:
- Main circle pipeline
- Main cross pipeline

Usage:
- Serves as main module for automatic grading pipeline
"""

from core_modules.base_preprocessing_modules import (
    automatic_warp_transformation_v2,
    image_uniformization,
    core_preprocessing_v2)
from core_modules.circles_modules import (
    extract_and_draw_contours, final_scoring)
from core_modules.cross_modules import (
    get_cross_answers,
    box_contour_handling,
    final_scoring_cross)


def main_circles_pipeline(answer_key, student_answer):
    """
    Main processing pipeline for handling circle-based answers.

    This function performs automatic grading by aligning and processing the answer key and student's answer sheets.
    It detects contours on the processed answer sheets and calculates the final score based on the contours detected.

    Parameters:
    - answer_key (ndarray): Image of the answer key sheet.
    - student_answer (ndarray): Image of the student's answer sheet.

    Returns:
    - stu_final_score (int): The calculated score for the student.
    - student_correction (ndarray): Image showing corrections (if any) on the student's answer sheet.
    - detected_total_questions (int): Total number of questions detected on the answer sheet.
    - detected_mistakes (int): Number of mistakes detected in the student's answers.
    """
    master_key = automatic_warp_transformation_v2(answer_key)
    student_sheet = automatic_warp_transformation_v2(student_answer)
    new_master, new_student = image_uniformization(master_key, student_sheet)
    processed_master = core_preprocessing_v2(new_master)
    processed_student = core_preprocessing_v2(new_student)
    # student_contours, student_contour_image = extract_and_draw_contours(processed_student)
    master_contours, _ = extract_and_draw_contours(processed_master)
    stu_final_score, student_correction, detected_total_questions, detected_mistakes = \
        final_scoring(new_student, processed_student, master_contours)

    return stu_final_score, student_correction, detected_total_questions, detected_mistakes


def main_cross_pipeline(answer_key, student_answer, model_instance):
    """
    Main processing pipeline for handling cross-based answers.

    This function processes the answer key and student's answer sheets for questions marked with crosses.
    It leverages a machine learning model to interpret cross marks, calculates the final score, and detects mistakes.

    Parameters:
    - answer_key (ndarray): Image of the answer key sheet.
    - student_answer (ndarray): Image of the student's answer sheet.
    - model_instance (object): A model instance used to interpret cross answers.

    Returns:
    - final_score (int): The calculated score for the student.
    - student_correction (ndarray): Image showing corrections (if any) on the student's answer sheet.
    - detected_total_questions (int): Total number of questions detected on the answer sheet.
    - detected_mistakes (int): Number of mistakes detected in the student's answers.
    """
    master_key = automatic_warp_transformation_v2(answer_key)
    student1 = automatic_warp_transformation_v2(student_answer)
    new_master, new_student = image_uniformization(master_key, student1)
    master_box_img, student_box_img = get_cross_answers(
        model_instance, new_master, new_student)
    processed_master = core_preprocessing_v2(master_box_img)
    processed_student = core_preprocessing_v2(student_box_img)
    master_contours, _, student_mistake_location = box_contour_handling(
        processed_master, processed_student)
    final_score, student_correction, detected_total_questions, detected_mistakes = \
        final_scoring_cross(new_student, master_contours, student_mistake_location)

    return final_score, student_correction, detected_total_questions, detected_mistakes
