import cv2
import matplotlib.pyplot as plt
import numpy as np

file_folder = "../../data/external/2_recognition/license_synthetic/sample_subset/"
sample1 = "data/external/2_recognition/license_synthetic/sample_subset/18-H-3396.png"
sample2 = "data/external/2_recognition/license_synthetic/license-plates/80-ZYY-26.png"
sample3 = "data/external/2_recognition/license_synthetic/license-plates/16-UML-08.png"

def read_image(filename: str):
    img =  cv2.cvtColor(cv2.imread(filename), cv2.COLOR_RGB2BGR)
    return img

def grayscale(image: np.ndarray):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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

def detect_contours(image):
    contour_img, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    character_bounding_boxes = list()

    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        image_width, image_height = image.shape[1], image.shape[0]
        
        if h > 0.9*image_height or h < 0.38*image_height:
            continue
        elif (h * w) < 0.01*(image_width * image_height):
            continue
        elif h / w < 1.1:
            continue
        else:
            cv2.rectangle(contour_img, (x,y), (x+w, y+h),(150,150,150),3)
            character_bounding_boxes.append((x,y,w,h))
    return contour_img, character_bounding_boxes

def snip_single_character(image, bounding_box):
    x, y, w, h = bounding_box
    return image[y:(y+h), x:(x+w)]

def snip_all_characters(image, bounding_boxes):
    return [snip_single_character(image, bounding_box) for bounding_box in bounding_boxes]

def standardize_snips(snips: list):
    snip_lst = snips.copy()
    for idx, img in enumerate(snip_lst):
        snip_lst[idx] = cv2.resize(img, (30,30)).astype('float32') / 255
    return snip_lst

def pipeline_single(filename: str, dilatekernel: tuple=(3,3), blurkernel: tuple=(5,5), div: int=25, gauss: bool=True):
    img = read_image(filename)
    grayscaled = grayscale(img)
    threshed = threshold_image(grayscaled)
    blurred = blur_image(threshed, blurkernel, div, gauss)
    dilated = dilate_image(blurred, dilatekernel)
    contours = detect_contours(dilated)[1]
    snips_lst = snip_all_characters(dilated, contours)
    return standardize_snips(snips_lst)
     
# Iterate for folder of images
def pipeline_bulk(folder: str, dilatekernel: tuple=(3,3), blurkernel: tuple=(5,5), div: int=25, gauss: bool=True):
    pass

if __name__ == '__main__':
    img = read_image(sample3)
    grayed = grayscale(img)
    threshed = threshold_image(grayed)
    blurred = blur_image(threshed, ksize=(7,7), div=25, gauss=True)
    dilated = dilate_image(blurred, ksize=(3,3), iters=1)
    characters = detect_contours(dilated.copy())[1]
    character = snip_single_character(img, characters[1])
    print(characters)

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(11,6), dpi=200)
    ax[0][0].imshow(img)
    ax[0][1].imshow(grayed, cmap='gray')
    ax[1][0].imshow(threshed, cmap='gray')
    ax[1][1].imshow(blurred, cmap='gray')
    ax[2][0].imshow(dilated, cmap='gray')
    ax[2][1].imshow(character,cmap='gray')
    plt.show()

    chars_lst = pipeline_single(sample3, dilatekernel=(3,3),blurkernel=(5,5), div=25, gauss=True)
    fig2, ax2 = plt.subplots(nrows=3, ncols=3, figsize=(11,6), dpi=200)
    for idx, ax in enumerate(ax2.flatten()):
        if idx > len(chars_lst)-1:
            break
        ax.imshow(chars_lst[idx], cmap='gray')
    plt.show()


    # (1000, 7, 30, 30) = data.shape
    #* batches of 7 in the CNN
    # (images, chracters, rows, cols)

    # train on a (1000, 
