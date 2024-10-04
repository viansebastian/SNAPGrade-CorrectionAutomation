from distutils.core import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np
import cv2

opencv_include = cv2.__file__.replace('__init__.py', '')
numpy_include = np.get_include()

extensions = [
    # Extension(
    #     "main_circles_processing", 
    #     ["main_circles_processing.pyx"], 
    #     language="c++",
    #     include_dirs=[opencv_include, numpy_include]
    #     )
    
    Extension(
        "base_preprocessing_modules_cy", 
        ["base_preprocessing_modules_cy.pyx"], 
        language="c++",
        include_dirs=[opencv_include, numpy_include]
        ),
    Extension(
        "circles_modules_cy", 
        ["circles_modules_cy.pyx"], 
        language="c++",
        include_dirs=[opencv_include, numpy_include]
        ),
    Extension(
        "circles_usage_cy", 
        ["circles_usage_cy.pyx"], 
        language="c++",
        include_dirs=[opencv_include, numpy_include]
        ),
    ]

setup(
    ext_modules=cythonize(extensions)
)