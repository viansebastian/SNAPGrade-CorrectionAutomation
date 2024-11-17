import cv2
import time
from main_pipelines import main_circles_pipeline, main_cross_pipeline
# from backup.main_circles_processing import main_circles_pipeline
# from circles_usage_cy import main_circles_pipeline

# answer1 = cv2.imread('inputs/circle_1/stu_good_light.jpg')
# master_sheet = cv2.imread('inputs/circle_1/master_circle_best.jpg')

answer_circle = cv2.imread('inputs/circle_2/student1_crop.jpg')
master_circle = cv2.imread('inputs/circle_2/master1_crop.jpg')

t1 = time.time()
stu_final_score_circle, stu_answer_key_circle, detected_total_questions_circle, detected_mistakes_circle = \
    main_circles_pipeline(master_circle, answer_circle)
t2 = time.time()

answer_cross = cv2.imread('C:/Users/vian8/Desktop/Tugas2/SNAPGRADE/inputs/cross/student2.jpg')
master_cross = cv2.imread('C:/Users/vian8/Desktop/Tugas2/SNAPGRADE/inputs/cross/master2.jpg')

t3 = time.time()
stu_final_score_cross, stu_answer_key_cross, detected_total_questions_cross, detected_mistakes_cross = \
    main_cross_pipeline(master_cross, answer_cross)
t4 = time.time()

print(f'circle time elapsed: {t2 - t1} s')
print(f'cross time elapsed: {t4 - t3} s')

cv2.imshow(f'Student Circle Results: {"{:.2f}".format(stu_final_score_circle)}', stu_answer_key_circle)
cv2.imshow(f'Student Cross Results: {"{:.2f}".format(stu_final_score_cross)}', stu_answer_key_cross)
cv2.waitKey(0)
cv2.destroyAllWindows()