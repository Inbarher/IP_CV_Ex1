import cv2
import numpy as np
from ex1_utils import imReadAndConvert


def gammaDisplay(img_path: str, rep: int) -> None:
    """
    GUI for gamma correction using OpenCV trackbars
    :param img_path: Path to the image
    :param rep: grayscale (1) or RGB(2)
    :return: None
    """
    # Load and normalize image [cite: 124, 133]
    img = imReadAndConvert(img_path, rep)

    # OpenCV uses BGR, but imReadAndConvert returns RGB or Gray
    if rep == 2:
        img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        img = (img * 255).astype(np.uint8)

    window_name = "Gamma Correction"
    cv2.namedWindow(window_name)

    def on_trackbar(val):
        # Convert integer 0-200 to float 0.0-2.0 [cite: 136, 137]
        gamma = val / 100.0
        # Power law transformation: I_out = I_in ^ gamma [cite: 124]
        # Normalize to float for math, then back to uint8 for display
        img_float = img.astype(np.float64) / 255.0
        corrected = np.power(img_float, gamma)
        cv2.imshow(window_name, (corrected * 255).astype(np.uint8))

    # Slider from 0 to 2 with resolution 0.01 (integer 0 to 200) [cite: 136]
    cv2.createTrackbar("Gamma", window_name, 100, 200, on_trackbar)

    # Initial display
    on_trackbar(100)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Simple test run
    gammaDisplay('testImg1.jpg', 2)