"""
Author  : Vian Sebastian B
Version : 1
Date    : 17/11/2024

"app.py "
This module contains the entry point for the service

Key Components:
- YOLO loading
- Circle processing pipeline
- Cross processing pipeline

Usage:
- Serves as the end-point for the service
"""

import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO 
from main_pipelines import main_circles_pipeline, main_cross_pipeline

app = Flask(__name__)
CORS(app)

model_instance = None

@app.before_first_request
def load_model():
    """
    Load the YOLO model before the first request.

    Description:
        Loads the YOLO model from the specified path and sets it as a global
        variable `model_instance`.

    Output:
        Sets a global `model_instance` with the YOLO model.
        Prints success or error messages to the console.
    """
    global model_instance
    try:
        model_instance = YOLO("C:/Users/vian8/Desktop/Tugas2/SNAPGRADE/model/yolo/best.pt")
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")

@app.route('/process-circles', methods=['POST'])
def process_circles():
    """
    Process the master sheet and student answer sheet using circle-based comparison.

    Parameters:
        - Request Body:
            - master_sheet (file): The image file of the master sheet.
            - student_answer (file): The image file of the student's answer sheet.

    Output:
        - JSON response containing:
            - score (int): Final score of the student.
            - total_questions (int): Total number of detected questions.
            - mistakes (list): Detected mistakes in the answers.
            - student_answer_key (str): Base64-encoded processed image.
    """
    master_sheet_file = request.files['master_sheet']
    student_answer_file = request.files['student_answer']
    
    print(f"Received master_sheet: {master_sheet_file.filename}")
    print(f"Received student_answer: {student_answer_file.filename}")

    # convert received img to cv2 format
    master_sheet_data = np.frombuffer(master_sheet_file.read(), np.uint8)
    master_sheet = cv2.imdecode(master_sheet_data, cv2.IMREAD_COLOR)  

    student_answer_data = np.frombuffer(student_answer_file.read(), np.uint8)
    student_answer = cv2.imdecode(student_answer_data, cv2.IMREAD_COLOR)  

    if master_sheet is None or student_answer is None:
        return jsonify({"error": "Failed to process the images."}), 400

    stu_final_score, stu_answer_key, detected_total_questions, detected_mistakes = main_circles_pipeline(master_sheet, student_answer)

    # convert the result image (stu_answer_key) from OpenCV to PNG format
    # then encode to base64 string
    _, buffer = cv2.imencode('.png', stu_answer_key)
    img_io = buffer.tobytes()
    img_base64 = base64.b64encode(img_io).decode('utf-8')

    # response dictionary (json)
    response = {
        'score': stu_final_score,
        'total_questions': detected_total_questions,
        'mistakes': detected_mistakes,
        'student_answer_key': img_base64  
    }

    return jsonify(response)  

@app.route('/process-cross', methods=['POST'])
def process_cross():
    """
    Process the master sheet and student answer sheet using cross-based comparison.

    Parameters:
        - Request Body:
            - master_sheet (file): The image file of the master sheet.
            - student_answer (file): The image file of the student's answer sheet.

    Output:
        - JSON response containing:
            - score (int): Final score of the student.
            - total_questions (int): Total number of detected questions.
            - mistakes (list): Detected mistakes in the answers.
            - student_answer_key (str): Base64-encoded processed image.
    """
    master_sheet_file = request.files['master_sheet']
    student_answer_file = request.files['student_answer']
    
    print(f"Received master_sheet: {master_sheet_file.filename}")
    print(f"Received student_answer: {student_answer_file.filename}")
    
    # convert received img to cv2 format
    master_sheet_data = np.frombuffer(master_sheet_file.read(), np.uint8)
    master_sheet = cv2.imdecode(master_sheet_data, cv2.IMREAD_COLOR)  

    student_answer_data = np.frombuffer(student_answer_file.read(), np.uint8)
    student_answer = cv2.imdecode(student_answer_data, cv2.IMREAD_COLOR)  

    if master_sheet is None or student_answer is None:
        return jsonify({"error": "Failed to process the images."}), 400

    stu_final_score, stu_answer_key, detected_total_questions, detected_mistakes = main_cross_pipeline(master_sheet, student_answer, model_instance)

    # convert the result image (stu_answer_key) from OpenCV to PNG format
    # then encode to base64 string
    _, buffer = cv2.imencode('.png', stu_answer_key)
    img_io = buffer.tobytes()
    img_base64 = base64.b64encode(img_io).decode('utf-8')

    # response dictionary (json)
    response = {
        'score': stu_final_score,
        'total_questions': detected_total_questions,
        'mistakes': detected_mistakes,
        'student_answer_key': img_base64  
    }

    return jsonify(response)  

if __name__ == '__main__':
    app.run(debug=True, port=5000)