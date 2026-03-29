import os
import sys
import matplotlib

base_python_path = r"C:\Users\Inbar\AppData\Local\Programs\Python\Python313"
os.environ['TCL_LIBRARY'] = os.path.join(base_python_path, "tcl", "tcl8.6")
os.environ['TK_LIBRARY'] = os.path.join(base_python_path, "tcl", "tk8.6")

matplotlib.use('TkAgg')
from ex1_utils import *
from gamma import gammaDisplay
# ------------------------------

def main():
    # 1. Check ID [cite: 14-18]
    print(f"Testing ID: {myID()}")

    # Set image path - make sure this file exists in your folder! [cite: 154]
    img_path = 'testImg1.jpg'

    if not os.path.exists(img_path):
        print(f"ERROR: General image file '{img_path}' not found in directory.")
        return

    # 2. Test Read and Convert [cite: 24-33]
    print("Testing imReadAndConvert...")
    im_gray = imReadAndConvert(img_path, 1)  # Gray
    im_rgb = imReadAndConvert(img_path, 2)  # RGB

    # 3. Test Display [cite: 40-47]
    print("Testing imDisplay (Close the window to continue)...")
    imDisplay(img_path, 1)

    # 4. Test YIQ Transformations [cite: 53-66]
    print("Testing YIQ Transformations...")
    yiq_img = transformRGB2YIQ(im_rgb)
    rgb_back = transformYIQ2RGB(yiq_img)

    # Check if conversion is consistent (Original vs Back-and-Forth)
    diff = np.abs(im_rgb - rgb_back).max()
    print(f"Max difference in RGB restoration: {diff:.6f}")

    # 5. Test Histogram Equalization [cite: 70-78]
    print("Testing Histogram Equalization...")
    # Test on RGB (will process Y channel internally) [cite: 79]
    img_eq, h_org, h_eq = histsogramEqualize(im_rgb)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1);
    plt.imshow(im_rgb);
    plt.title("Original")
    plt.subplot(1, 2, 2);
    plt.imshow(img_eq);
    plt.title("Equalized")
    plt.show()

    # 6. Test Quantization [cite: 90-100]
    print("Testing Quantization (nQuant=4, nIter=10)...")
    # This will also plot the error graph [cite: 119]
    q_imgs, errors = quantizeImage(im_rgb, nQuant=4, nIter=10)

    if q_imgs:
        plt.figure()
        plt.imshow(q_imgs[-1])
        plt.title("Final Quantized Image")
        plt.show()

    # 7. Test Gamma Display [cite: 123-135]
    print("Testing Gamma Correction GUI...")
    gammaDisplay(img_path, 2)


if __name__ == '__main__':
    main()