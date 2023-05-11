import cv2
import matplotlib.pyplot as plt
import numpy as np

raw_images_folder = './data/raw/'
sample_image = './data/raw/17-SAC-73.png'


def read_image(filename: str) -> np.ndarray:
    '''Reads a specified image and converts to GBR color scheme used by OpenCV.

    Args:
    ----
        filename (str): Name of an individual unprocessed image.

    Returns:
    -------
        [np.ndarray]: Array of the image data of shape
                      (height, width, channels)
    '''
    try:
        img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imread(filename)
    return img


def grayscale(image: np.ndarray) -> np.ndarray:
    '''Converts image to a single grayscale channel.

    Args:
    ----
        image (np.ndarray): Image in an RGB or BGR color scheme.

    Returns:
    -------
        [np.ndarray]: Image in a single grayscale channel.
                      Expected shape: (height, width, 1)
    '''
    try:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    except Exception:
        return image


def threshold_image(image: np.ndarray) -> np.ndarray:
    '''Applies a threshold to the supplied image.

    Args:
    ----
        image (np.ndarray): Expecting a grayscaled image,
                            but can be used more generally.

    Returns:
    -------
        [np.ndarray]: An image array where all values are either 0 or 255
    '''
    threshold_image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU, cv2.THRESH_BINARY_INV)[1]
    return cv2.bitwise_not(threshold_image)


def display_hist(image, channel: int = 0):
    hist_values = cv2.calcHist([image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    plt.plot(hist_values)


def blur_image(image: np.ndarray, ksize: tuple, div: int, gauss: bool):
    # ! div=1 removes State section! (unintended feature)
    if gauss:
        kernel = np.ones(shape=ksize, dtype=np.float32) / div
        gauss_blurred = cv2.filter2D(image, -1, kernel)
        return gauss_blurred
    else:
        median_blurred = cv2.medianBlur(image, ksize[0])
        return median_blurred


def detect_edges(image):
    med_val = np.median(image)
    lower = int(max(0, 0.07 * med_val))
    upper = int(min(255, 1.3 * med_val))
    return cv2.Canny(image=image, threshold1=lower, threshold2=upper + 100)


def dilate_image(image, ksize: tuple = (5, 5), iters: int = 1, erode=True):
    kernel_dilation = np.ones(ksize, dtype=np.uint8)
    if erode:
        dilated = cv2.erode(image, kernel_dilation, iterations=iters)
    else:
        dilated = cv2.dilate(image, kernel_dilation, iterations=iters)
    return dilated


def pipeline_single(
    filename: str,
    dilatekernel: tuple = (3, 3),
    blurkernel: tuple = (5, 5),
    div: int = 25,
    gauss: bool = True,
):
    img = read_image(filename)
    grayscaled = grayscale(img)
    threshed = threshold_image(grayscaled)
    dilated = dilate_image(threshed, dilatekernel)
    blurred = blur_image(dilated, blurkernel, div, gauss)
    return blurred


def save_image(filename, image):
    img = np.array(image)
    print(img.shape)
    cv2.imwrite(filename, img)


if __name__ == '__main__':
    pass
