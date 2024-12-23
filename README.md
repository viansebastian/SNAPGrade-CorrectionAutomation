﻿# SNAPGrade: System for Numeric and Alphabetic Processing in Grading and Dynamic Evaluation

***This repository contains the backend service for SNAPGrade.***

SNAPGrade is an AI-powered mobile application designed to assist educators in grading multiple-choice answer sheets efficiently. This smart image processing solution streamlines the evaluation process, eliminating the need for specialized hardware. With SNAPGrade, educators can save valuable time and focus more on teaching.

This project is funded and supported by the Department of Computer Science and Electronics, Universitas Gadjah Mada, Yogyakarta, Indonesia.

---

SNAPGrade works by this paradigm: 

<div align="center"> <img src="docs\snapgrade-flowchart.jpg?raw=true" alt="Circle Master Key" width="750"></div>


1. A answered sheet (master key) is uploaded
2. The student's answer is uploaded
3. The template is inputed, between circles or cross
3. Respective processing strategy is performed, and returns the final score as well as corrections.

### Module Explanations: 
- The Circles are processed with pure digital image processing techniques, employing adaptive the blob detection algorithm.
- The Cross are processed using a trained YOLOv8 model, focusing on localizing 'X' symbols, continued with further custom processing.
- The Preprocessing applied is robust as it employs adaptive parameterization, allowing it to handle various conditions of input images. 

### How SNAPGrade Benefits Educators
1. **Time Efficiency**: Automates the grading process, drastically reducing time spent on manual corrections.
      - Time for Circle Correction (demo): 1.92 seconds
      - Time for Cross Correction (demo): 3.64 seconds
2. **Cost-Effective**: Requires only a smartphone, removing the need for costly hardware or specialized scanners.
3. **Ease of Use**: Intuitive interface designed with educators in mind

### Limitations
1. The input must already be the region of interest (cannot be inputted with full page)
2. Image must be taken from a flat surface (cannot be bent)
3. Requires internet (backend service is deployed to [Railway](https://railway.app))
4. Although robust, a broken image may not work (too bright, too noisy, too dark, too much differences in lighting)
   
### Next Iterations: 
1. Backend to be developed locally, enabling offline use 
2. Employing databases
3. Development of better UX


<div align='center'>

## Example Circle Input

*Master Key (Left) and Student Answer (Right) Sample Input*

<div align="center"> <img src="docs/master_circle_sample.png?raw=true" alt="Circle Master Key" width="202"> <img src="docs/student_circle_sample.png?raw=true" alt="Circle Student Answer" width="200"> </div>

## Circle Correction Output

<div align="center"> <img src="docs/result_circle_sample.png?raw=true" alt="Circle Result" width="200"> </div>

## Example Cross Input

*Master Key (Left) and Student Answer (Right) Sample Input*

<div align="center"> <img src="docs/master_cross_sample.png?raw=true" alt="Cross Master Key" width="400"> <img src="docs/student_cross_sample.png?raw=true" alt="Cross Student Answer" width="416"> </div>

## Cross Correction Output

<div align="center"> <img src="docs/result_cross_sample.png?raw=true" alt="Cross Result" width="400"> </div>


</div>

## Demo Video 

See SNAPGrade in action:
[View Demo](https://drive.google.com/file/d/1pswuRX2sY8vQ05sCXV2bYxMsEbINYX3s/view?usp=sharing)

## Frontend Repo

Special thanks to [Maulana Arya](https://github.com/MaulanaArya30) for his amazing frontend work!

Check out the mobile app here: [SNAPGrade-MobileApp](https://github.com/MaulanaArya30/SNAPGrade_app)
