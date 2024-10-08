import cv2
import time
from main_pipelines import main_circles_pipeline
# from backup.main_circles_processing import main_circles_pipeline
# from circles_usage_cy import main_circles_pipeline

# answer1 = cv2.imread('inputs/circle_1/stu_good_light.jpg')
# master_sheet = cv2.imread('inputs/circle_1/master_circle_best.jpg')

answer1 = cv2.imread('inputs/circle_2/student1_crop.jpg')
master_sheet = cv2.imread('inputs/circle_2/master1_crop.jpg')

t1 = time.time()
stu_final_score, stu_answer_key, detected_total_questions, detected_mistakes = main_circles_pipeline(master_sheet, answer1)
t2 = time.time()

print(f'time elapsed: {t2-t1} s')
# print('Detected Questions: ', detected_total_questions)
cv2.imshow(f'Student Results: {"{:.2f}".format(stu_final_score)}', stu_answer_key)
cv2.waitKey(0)
cv2.destroyAllWindows()