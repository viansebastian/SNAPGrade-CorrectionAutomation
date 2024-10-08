import cv2
import time
from base_preprocessing_modules import ( automatic_warp_transformation, image_uniformization )
from circles_modules import ( core_preprocessing_v2, extract_and_draw_contours, final_scoring  )

def main_circles_pipeline(answer_key, student_answer): 
    student_sheet = automatic_warp_transformation(student_answer)
    master_key = automatic_warp_transformation(answer_key)
    new_master, new_student = image_uniformization(master_key, student_sheet)
    processed_master = core_preprocessing_v2(new_master)
    processed_student = core_preprocessing_v2(new_student)
    student_contours, student_contour_image = extract_and_draw_contours(processed_student)
    master_contours, master_contour_image = extract_and_draw_contours(processed_master)
    stu_final_score, stu_answer_key = final_scoring(new_student, processed_student, master_contours)
    
    return stu_final_score, stu_answer_key
