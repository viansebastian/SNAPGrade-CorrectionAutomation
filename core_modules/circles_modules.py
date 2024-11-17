import numpy as np
import cv2 
from core_modules.base_preprocessing_modules import soft_morph_open

def draw_full_contours(contours, cont_image, radius = 7):
  '''Draw Full Circles'''
  for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
      
      # draw filled circle at contour center
      cv2.circle(cont_image, (cX, cY), radius, (0, 255, 0), -1)

  return cont_image

def extract_and_draw_contours(image):
  contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  unique_values = []
  for columns in image:
    for pixel in columns:
      if pixel not in unique_values:
        unique_values.append(pixel)

  contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  contour_image = draw_full_contours(contours, contour_image)

  return contours, contour_image

def extract_and_draw_circle_contours(image):
  contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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
  '''Final Score Calculation'''
  
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
  student_correction = draw_full_contours(master_contours, student_correction)

  return final_score, student_correction, total_questions, mistakes