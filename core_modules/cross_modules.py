import numpy as np
import cv2
import torch
from ultralytics import YOLO 
from core_modules.base_preprocessing_modules import soft_morph_open

print(cv2.__version__)
print(np.__version__)

def model():
    try:
        model = YOLO("C:/Users/vian8/Desktop/Tugas2/SNAPGRADE/model/yolo/best.pt")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def yolo_catch_image(model, image): 
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
    Computes the maximum width and height from master and student bounding box coordinates.

    Args:
        master_coords (torch.Tensor or numpy.ndarray): Bounding box coordinates for the master image (center_x, center_y, width, height).
        student_coords (torch.Tensor or numpy.ndarray): Bounding box coordinates for the student image (center_x, center_y, width, height).

    Returns:
        tuple: Maximum width and maximum height across both sets of coordinates.
    """
    max_width = 0
    max_height = 0

    # Combine both sets of coordinates
    all_coords = torch.cat([master_coords, student_coords], dim=0) if isinstance(master_coords, torch.Tensor) else np.vstack((master_coords, student_coords))
    
    # Traverse all bounding box dimensions
    for coord in all_coords:
        # Convert tensor to NumPy array if needed
        if isinstance(coord, torch.Tensor):
            coord = coord.cpu().numpy()
        _, _, width, height = coord.astype(int)
        
        # Update maximum width and height
        max_width = max(max_width, width)
        max_height = max(max_height, height)

    # return max(max_width, max_height)
    return int((max_width + max_height) / 2)

def mark_ans_box(image, coords, shape="box", size=10, color=(0, 255, 0), thickness=-1):
    """
    Draw a fixed-size filled box or circle at the center of each detected box.

    Args:
        image (numpy.ndarray): The input image where the shapes will be drawn.
        boxes (object): The detected boxes (used for validation, if needed).
        coords (torch.Tensor or numpy.ndarray): Box coordinates (center x, center y, width, height).
        shape (str): The shape to draw ("box" or "circle"). Default is "box".
        size (int): The size (side length or radius) of the shape. Default is 10.
        color (tuple): The color of the shape in (B, G, R). Default is green.
        thickness (int): Thickness of the shape. Use -1 for filled shapes. Default is -1.

    Returns:
        numpy.ndarray: The image with the shapes drawn on it.
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
    Draws filled boxes inside each white region in a binary image.

    Args:
        image (numpy.ndarray): Grayscale input image with white regions on a black background.

    Returns:
        numpy.ndarray: Image with filled boxes drawn inside each white region.
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
    Draws filled boxes inside each white region in a binary image.

    Args:
        image (numpy.ndarray): Grayscale input image with white regions on a black background.

    Returns:
        numpy.ndarray: Image with filled boxes drawn inside each white region.
    """
    # Convert to binary if not already
    # _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    input_img = image.copy()
    
    # Find contours
    contours, _ = cv2.findContours(input_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert to color for drawing
    result_image = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    
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
        cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), -1)  # Green filled box

    return contours, result_image

def draw_box_fill_space_overlap(master_contours, student_contours, image_shape, overlap_threshold=0.1):
    """
    Draws contours with conditions:
    - Regions with >=10% overlap between student and master contours are filled green.
    - Non-overlapping or <10% overlap contours from student_contours are left white.

    Args:
        master_contours (list): Contours from the master (reference) image.
        student_contours (list): Contours from the student (target) image.
        image_shape (tuple): Shape of the new image (height, width, channels).
        overlap_threshold (float): Minimum overlap ratio to classify as overlapping (default: 10%).

    Returns:
        numpy.ndarray: Image with processed contours.
    """
    # Create a blank image for the result
    result_img = np.zeros(image_shape, dtype=np.uint8)

    # Create masks for master and student contours
    master_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    # student_mask = np.zeros(image_shape[:2], dtype=np.uint8)

    # Draw the master contours on the master mask
    cv2.drawContours(master_mask, master_contours, -1, 255, thickness=cv2.FILLED)

    for contour in student_contours:
        # Create an individual mask for the current student contour
        single_student_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.drawContours(single_student_mask, [contour], -1, 255, thickness=cv2.FILLED)

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
    master_pred, master_box, master_coords = yolo_catch_image(model, master_image)
    student_pred, student_box, student_coords = yolo_catch_image(model, student_image)
    
    box_size = get_max_width_height(master_coords, student_coords)
    
    master_img = mark_ans_box(master_image, master_coords, shape='box', size=box_size, color=(0, 255, 0), thickness=-1)
    student_img = mark_ans_box(student_image, student_coords, shape='box', size=box_size, color=(0, 255, 0), thickness=-1)
    
    return master_img, student_img

def box_contour_handling(master_image, student_image):
    master_contours, master_output = draw_filled_boxes_fetch_contours(master_image)
    student_contours, student_output = draw_filled_boxes_fetch_contours(student_image)
    
    shape = (student_image.shape[0], student_image.shape[1], 3)
    stu_mistake_loc_cont_img = draw_box_fill_space_overlap(master_contours, student_contours, shape)
    stu_mistake_loc_cont_img = cv2.cvtColor(stu_mistake_loc_cont_img, cv2.COLOR_BGR2GRAY)
    _, result_img = cv2.threshold(stu_mistake_loc_cont_img, 254, 255, cv2.THRESH_BINARY)
    stu_mistake_loc_final = soft_morph_open(result_img)
    
    return master_contours, student_contours, stu_mistake_loc_final

def final_scoring_cross(student_img, master_contours, student_mistake_loc): 
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