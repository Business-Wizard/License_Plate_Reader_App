import cv2
import matplotlib.pyplot as plt
import numpy as np

file_folder = "../../data/external/2_recognition/license_synthetic/sample_subset/"
sample1 = "data/external/2_recognition/license_synthetic/sample_subset/18-H-3396.png"
sample2 = "data/external/2_recognition/license_synthetic/license-plates/80-ZYY-26.png"
sample3 = "data/external/2_recognition/license_synthetic/license-plates/16-UML-08.png"

def read_image(filename: str):
    try:
        img =  cv2.cvtColor(cv2.imread(filename), cv2.COLOR_RGB2BGR)
    except:
        img =  cv2.imread(filename)
    return img

def grayscale(image: np.ndarray):
    try:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    except:
        return image

def threshold_image(image):
    threshold_image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU, cv2.THRESH_BINARY_INV)[1]
    return cv2.bitwise_not(threshold_image)

def display_hist(image, channel: int=0):
    hist_values = cv2.calcHist([image], channels=[0], mask=None, histSize=[256], ranges=[0,256])
    plt.plot(hist_values)

def blur_image(image: np.ndarray, ksize: tuple, div: int, gauss: bool):
    #! div=1 removes State section! (unintended feature)
    if gauss == True:
        kernel = np.ones(shape=ksize, dtype=np.float32) / div
        gauss_blurred = cv2.filter2D(image, -1, kernel)
        return gauss_blurred
    else:
        median_blurred = cv2.medianBlur(image, ksize[0])
        return median_blurred

def detect_edges(image):
    med_val = np.median(image)
    lower = int(max(0, 0.07*med_val) )
    upper = int(min(255, 1.3*med_val) )
    return cv2.Canny(image=image, threshold1=lower, threshold2=upper+100)

def dilate_image(image, ksize: tuple=(5,5), iters: int=1):
    kernel_dilation = np.ones(ksize, dtype=np.uint8)
    dilated = cv2.dilate(image, kernel_dilation, iterations=iters)
    return dilated

def pipeline_single(filename: str, dilatekernel: tuple=(3,3), blurkernel: tuple=(5,5), div: int=25, gauss: bool=True):
    img = read_image(filename)
    grayscaled = grayscale(img)
    threshed = threshold_image(grayscaled)
    blurred = blur_image(threshed, blurkernel, div, gauss)
    dilated = dilate_image(blurred, dilatekernel)
    return dilated
    
# Iterate for folder of images
def pipeline_bulk(folder: str, dilatekernel: tuple=(3,3), blurkernel: tuple=(5,5), div: int=25, gauss: bool=True):
    #! implemented in makedataset script
    pass

if __name__ == '__main__':
    img = read_image(sample3)
    grayed = grayscale(img)
    threshed = threshold_image(grayed)
    blurred = blur_image(threshed, ksize=(7,7), div=25, gauss=True)
    dilated = dilate_image(blurred, ksize=(3,3), iters=1)

    # fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(11,6), dpi=200)
    # ax[0][0].imshow(img)
    # ax[0][1].imshow(grayed, cmap='gray')
    # ax[1][0].imshow(threshed, cmap='gray')
    # ax[1][1].imshow(blurred, cmap='gray')
    # ax[2][0].imshow(dilated, cmap='gray')
    # plt.show()



    # (1000, 7, 30, 30) = data.shape
    #? batches of 7 in the CNN?
    # (images, chracters, rows, cols)

    # train on a (1000, 
