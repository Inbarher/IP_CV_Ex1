import numpy as np
import cv2
from typing import List
import matplotlib.pyplot as plt


def myID() -> int:
    """
    Return my ID
    :return: int
    """
    # [cite: 18]
    return 214450371


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: grayscale (1) or RGB (2)
    :return: The image np array normalized to [0,1]
    """
    # Read image using OpenCV (loads as BGR) [cite: 34]
    img = cv2.imread(filename)
    if img is None:
        return np.array([])

    if representation == 1:
        # Convert to Grayscale [cite: 30]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif representation == 2:
        # Convert BGR to RGB [cite: 30, 47]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1] and convert to float [cite: 33]
    return img.astype(np.float64) / 255.0


def imDisplay(filename: str, representation: int) -> None:
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: grayscale (1) or RGB(2)
    :return: None
    """
    # Load image using the previous function [cite: 38]
    img = imReadAndConvert(filename, representation)

    plt.figure()
    if representation == 1:
        # Display as grayscale [cite: 46]
        plt.imshow(img, cmap='gray')
    else:
        # Display as RGB [cite: 47]
        plt.imshow(img)
    plt.show()


def transformRGB2YIQ(imRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # Transformation matrix from documentation [cite: 51]
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.275, -0.321],
                             [0.212, -0.523, 0.311]])

    # Vectorial multiplication using dot product [cite: 66, 145]
    return imRGB.dot(yiq_from_rgb.T)


def transformYIQ2RGB(imYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # Inverse matrix for YIQ to RGB conversion [cite: 64]
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.275, -0.321],
                             [0.212, -0.523, 0.311]])
    rgb_from_yiq = np.linalg.inv(yiq_from_rgb)

    # Vectorial multiplication [cite: 66]
    return imYIQ.dot(rgb_from_yiq.T)


def histsogramEqualize(imOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Equalizes the histogram of an image
    :param imOrig: Original image
    :return: (imgEq, historg, histEQ)
    """
    # Handle RGB by processing only the Y channel [cite: 79]
    is_rgb = len(imOrig.shape) == 3
    if is_rgb:
        yiq = transformRGB2YIQ(imOrig)
        work_img = yiq[:, :, 0]
    else:
        work_img = imOrig

    # Step 1: Calculate image histogram (scaled to 0-255) [cite: 82, 88]
    img_255 = (work_img * 255).astype(np.uint8)
    hist_org, _ = np.histogram(img_255.flatten(), bins=256, range=(0, 256))

    # Step 2: Calculate normalized Cumulative Sum [cite: 83]
    cumsum = hist_org.cumsum()
    cumsum_norm = cumsum / cumsum[-1]

    # Step 3: Create LookUpTable (LUT) [cite: 84]
    lut = (cumsum_norm * 255).astype(np.uint8)

    # Step 4: Replace intensities [cite: 85]
    img_eq_255 = lut[img_255]
    img_eq = img_eq_255.astype(np.float64) / 255.0

    # Final histogram [cite: 89]
    hist_eq, _ = np.histogram(img_eq_255.flatten(), bins=256, range=(0, 256))

    if is_rgb:
        yiq[:, :, 0] = img_eq
        img_eq = transformYIQ2RGB(yiq)

    return img_eq, hist_org, hist_eq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
    Quantized an image in to nQuant colors
    :param imOrig: The original image (RGB or Gray scale)
    :param nQuant: Number of colors to quantize the image to
    :param nIter: Number of optimization loops
    :return: (List [qImage_i], List [error_i])
    """
    # Handle RGB [cite: 101]
    is_rgb = len(imOrig.shape) == 3
    if is_rgb:
        yiq = transformRGB2YIQ(imOrig)
        y_channel = yiq[:, :, 0]
    else:
        y_channel = imOrig

    y_255 = (y_channel * 255).astype(np.uint8)
    hist, _ = np.histogram(y_255, bins=256, range=(0, 256))

    # Initialize borders z with equal number of pixels [cite: 115]
    cumsum = hist.cumsum()
    step = cumsum[-1] // nQuant
    z = np.zeros(nQuant + 1, dtype=int)
    for i in range(1, nQuant):
        z[i] = np.searchsorted(cumsum, i * step)
    z[nQuant] = 255  # [cite: 106]

    q = np.zeros(nQuant)
    images = []
    errors = []

    for _ in range(nIter):
        # Find optimal q (weighted means of segments) [cite: 107, 112]
        for i in range(nQuant):
            range_vals = np.arange(z[i], z[i + 1] + 1)
            h_vals = hist[z[i]: z[i + 1] + 1]
            if h_vals.sum() > 0:
                q[i] = (range_vals * h_vals).sum() / h_vals.sum()

        # Calculate MSE error [cite: 117, 118]
        error = 0
        for i in range(nQuant):
            range_vals = np.arange(z[i], z[i + 1] + 1)
            h_vals = hist[z[i]: z[i + 1] + 1]
            error += np.sum(((q[i] - range_vals) ** 2) * h_vals)
        errors.append(error)

        # Update quantized image [cite: 100]
        lut = np.zeros(256)
        for i in range(nQuant):
            lut[z[i]: z[i + 1] + 1] = q[i]

        q_img_y = lut[y_255] / 255.0
        if is_rgb:
            tmp_yiq = yiq.copy()
            tmp_yiq[:, :, 0] = q_img_y
            images.append(transformYIQ2RGB(tmp_yiq))
        else:
            images.append(q_img_y)

        # Update borders z [cite: 105, 112]
        old_z = z.copy()
        for i in range(1, nQuant):
            z[i] = (q[i - 1] + q[i]) / 2

        if np.array_equal(old_z, z):  # Convergence check
            break

    plt.plot(errors)  # [cite: 119]
    plt.show()
    return images, errors