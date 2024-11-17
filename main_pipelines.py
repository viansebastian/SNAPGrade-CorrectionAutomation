from core_modules.base_preprocessing_modules import ( automatic_warp_transformation, image_uniformization, core_preprocessing_v2 )
from core_modules.circles_modules import ( extract_and_draw_contours, final_scoring )
from core_modules.cross_modules import ( get_cross_answers, box_contour_handling, final_scoring_cross)

def main_circles_pipeline(answer_key, student_answer): 
    master_key = automatic_warp_transformation(answer_key)
    student_sheet = automatic_warp_transformation(student_answer)
    new_master, new_student = image_uniformization(master_key, student_sheet)
    processed_master = core_preprocessing_v2(new_master)
    processed_student = core_preprocessing_v2(new_student)
    # student_contours, student_contour_image = extract_and_draw_contours(processed_student)
    master_contours, _ = extract_and_draw_contours(processed_master)
    stu_final_score, student_correction, detected_total_questions, detected_mistakes = \
        final_scoring(new_student, processed_student, master_contours)
    
    return stu_final_score, student_correction, detected_total_questions, detected_mistakes

def main_cross_pipeline(answer_key, student_answer, model_instance):
    master_key = automatic_warp_transformation(answer_key)
    student1 = automatic_warp_transformation(student_answer)
    new_master, new_student = image_uniformization(master_key, student1)
    master_box_img, student_box_img = get_cross_answers(model_instance, new_master, new_student)
    processed_master = core_preprocessing_v2(master_box_img)
    processed_student = core_preprocessing_v2(student_box_img)
    master_contours, _, student_mistake_location = box_contour_handling(processed_master, processed_student)
    final_score, student_correction, detected_total_questions, detected_mistakes = \
        final_scoring_cross(new_student, master_contours, student_mistake_location)

    return final_score, student_correction, detected_total_questions, detected_mistakes