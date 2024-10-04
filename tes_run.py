import cv2
# from main_circles_processing import main_circles_pipeline
from circles_usage_cy import main_circles_pipeline

# answer1 = cv2.imread('inputs/circle_1/stu_good_light.jpg')
# master_sheet = cv2.imread('inputs/circle_1/master_circle_best.jpg')

answer1 = cv2.imread('inputs/circle_2/student1_crop.jpg')
master_sheet = cv2.imread('inputs/circle_2/master1_crop.jpg')

stu_final_score, stu_answer_key = main_circles_pipeline(master_sheet, answer1)

cv2.imshow(f'Student Results: {"{:.2f}".format(stu_final_score)}', stu_answer_key)
cv2.waitKey(0)
cv2.destroyAllWindows()