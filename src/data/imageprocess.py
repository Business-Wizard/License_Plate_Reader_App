import logging
import pathlib
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

RAW_IMAGES_DIR: Path = pathlib.Path().cwd() / 'data/raw/'
SAMPLE_IMAGE: Path = RAW_IMAGES_DIR / '17-SAC-73.png'


def read_image(filename: Path) -> np.ndarray:
    try:
        img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001
        img = cv2.imread(filename)
    return img


def grayscale(image: np.ndarray) -> np.ndarray:
    try:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    except Exception:  # noqa: BLE001
        return image


def threshold_image(image: np.ndarray) -> np.ndarray:
    threshold_image: np.ndarray = cv2.threshold(
        src=image,
        thresh=0,
        maxval=255,
        type=cv2.THRESH_OTSU,
        dst=cv2.THRESH_BINARY_INV,
    )[1]
    return cv2.bitwise_not(threshold_image)


def display_hist(image: np.ndarray, _channel: int = 0) -> None:
    hist_values = cv2.calcHist([image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    plt.plot(hist_values)


def blur_image(image: np.ndarray, ksize: tuple[int, ...], *, div: int, gauss: bool) -> np.ndarray:
    # ! div=1 removes State section! (unintended feature)
    if gauss:
        kernel = np.ones(shape=ksize, dtype=np.float32) / div
        return cv2.filter2D(image, -1, kernel)
    return cv2.medianBlur(image, ksize[0])


def detect_edges(image: np.ndarray):
    med_val = np.median(image)
    lower = int(max(0, 0.07 * med_val))
    upper = int(min(255, 1.3 * med_val))
    return cv2.Canny(image=image, threshold1=lower, threshold2=upper + 100)


def dilate_image(image, ksize: tuple = (5, 5), iters: int = 1, *, erode: bool = True) -> np.ndarray:
    kernel_dilation = np.ones(ksize, dtype=np.uint8)
    if erode:
        dilated = cv2.erode(image, kernel_dilation, iterations=iters)
    else:
        dilated = cv2.dilate(image, kernel_dilation, iterations=iters)
    return dilated


def pipeline_single(
    filepath: Path,
    dilatekernel: tuple = (3, 3),
    blurkernel: tuple = (5, 5),
    div: int = 25,
    *,
    gauss: bool = True,
):
    img = read_image(filepath)
    grayscaled = grayscale(img)
    threshed = threshold_image(grayscaled)
    dilated = dilate_image(threshed, dilatekernel)
    return blur_image(image=dilated, ksize=blurkernel, div, gauss)


def save_image(filename, image):
    img = np.array(image)
    logging.info(img.shape)
    cv2.imwrite(filename, img)


if __name__ == '__main__':
    pass
