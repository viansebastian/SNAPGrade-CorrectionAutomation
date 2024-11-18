"""
Author  : Vian Sebastian B
Version : 1
Date    : 17/11/2024

"main.py "
This module contains circle-specific handling functions

Key Components:
- Blob detection and handling
- Scoring computation

Usage:
- Serves as circle-specific handling module
"""

import cv2
import time
from main_pipelines import main_circles_pipeline, main_cross_pipeline
from ultralytics import YOLO 

model_instance = YOLO("C:/Users/vian8/Desktop/Tugas2/SNAPGRADE/model/yolo/best.pt")

# answer_circle = cv2.imread('inputs/circle_2/student1_crop.jpg')
# master_circle = cv2.imread('inputs/circle_2/master1_crop.jpg')

# t1 = time.time()
# stu_final_score_circle, stu_answer_key_circle, detected_total_questions_circle, detected_mistakes_circle = \
#     main_circles_pipeline(master_circle, answer_circle)
# t2 = time.time()

answer_cross = cv2.imread('C:/Users/vian8/Desktop/Tugas2/SNAPGRADE/inputs/cross_data_cropped/9.jpg')
master_cross = cv2.imread('C:/Users/vian8/Desktop/Tugas2/SNAPGRADE/inputs/cross_data_cropped/8.jpg')

t3 = time.time()
stu_final_score_cross, stu_answer_key_cross, detected_total_questions_cross, detected_mistakes_cross = \
    main_cross_pipeline(master_cross, answer_cross, model_instance)
t4 = time.time()

# print(f'circle time elapsed: {t2 - t1} s')
print(f'cross time elapsed: {t4 - t3} s')

# cv2.imshow(f'Student Circle Results: {"{:.2f}".format(stu_final_score_circle)}', stu_answer_key_circle)
cv2.imshow(f'Student Cross Results: {"{:.2f}".format(stu_final_score_cross)}', stu_answer_key_cross)
cv2.waitKey(0)
cv2.destroyAllWindows()