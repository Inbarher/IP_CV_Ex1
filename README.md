# Computer Vision & Image Processing - Exercise 1

**Student Name:** Inbar Hermos
**Student ID:** 2144*****
**GitHub Username:** Inbarher

## System Information
* **Python Version:** 3.13
* **Operating System:** Windows 11
* **Libraries used:** NumPy, OpenCV (cv2), Matplotlib

## Submitted Files
* **ex1_utils.py**: The main utility file containing core image processing functions (Color conversions, Histogram Equalization, and Quantization).
* **gamma.py**: Implementation of the interactive Gamma Correction GUI using OpenCV trackbars.
* **ex1_main.py**: A test script provided to run and verify the functionality of all implemented parts.
* **testImg1.jpg & testImg2.jpg**: Original test images used for evaluating the algorithms.

## Implemented Functions Descriptions
* **myID()**: Returns the student's ID as an integer.
* **imReadAndConvert(filename, representation)**: Reads an image file and converts it to Grayscale (1) or RGB (2) representation, normalized to the range [0, 1].
* **imDisplay(filename, representation)**: Utilizes `imReadAndConvert` to display a loaded image using Matplotlib.
* **transformRGB2YIQ(imRGB)**: Converts an RGB image to the YIQ color space using vectorial matrix multiplication.
* **transformYIQ2RGB(imYIQ)**: Converts an image from YIQ color space back to RGB.
* **histogramEqualize(imOrig)**: Performs histogram equalization on a grayscale or RGB image (via the Y channel) to improve contrast.
* **quantizeImage(imOrig, nQuant, nIter)**: Performs optimal image quantization by minimizing the Mean Squared Error (MSE) over a set of iterations.
* **gammaDisplay(img_path, rep)**: Opens a GUI window with a trackbar to adjust the gamma correction value of an image dynamically.

## Execution Instructions
1. Ensure all libraries (numpy, opencv-python, matplotlib) are installed.
2. Place the code files and test images in the same directory.
3. Run the test script using the command: `python ex1_main.py`.

