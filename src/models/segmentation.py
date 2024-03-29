import logging

import cv2
import matplotlib.pyplot as plt


def detect_contours(image):
    logging.info('Detecting contours')
    logging.info(f'Shape of image for contour detection: {image.shape}')
    img = image.copy()
    try:
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except Exception:  # noqa: BLE001
        logging.warning('provided image ')
        contour_img, contours, hierarchy = cv2.findContours(
            img,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    character_bounding_boxes = []

    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        _image_width, image_height = image.shape[1], image.shape[0]

        if _is_character := 0.40 < h / image_height < 0.9:
            cv2.rectangle(img, (x, y), (x + w, y + h), (150, 150, 150), 1)
            character_bounding_boxes.append((x, y, w, h))

    return img, character_bounding_boxes


def snip_single_character(image, bounding_box):
    x, y, w, h = bounding_box
    return image[y : (y + h), x : (x + w)]


def snip_all_characters(image, bounding_boxes):
    return [snip_single_character(image, bounding_box) for bounding_box in bounding_boxes]


def standardize_snips(snips: list):
    snip_lst = snips.copy()
    for idx, img in enumerate(snip_lst):
        snip_lst[idx] = cv2.resize(img, (30, 30)).astype('float32') / 255
    return snip_lst


def segment_image(image):
    contours = detect_contours(image)[1]
    snips_lst = snip_all_characters(image, contours)
    return standardize_snips(snips_lst)


if __name__ == '__main__':
    filepath = '/home/joseph/Documents/10_EDUCATION/10_galvanize/'
    '51_capstones/2_license_plates/license_plate_recognition/data/'
    'processed/2_recognition/train_set/43ir353.png'
    image = cv2.imread(filepath)
    logging.info(image.shape)
    plt.imshow(image, cmap='gray')
    plt.show()

    chars_lst = segment_image(cv2.imread(filepath, 0))
    fig2, ax2 = plt.subplots(nrows=3, ncols=3, figsize=(11, 6), dpi=200)
    for idx, ax in enumerate(ax2.flatten()):
        if idx > len(chars_lst) - 1:
            break
        logging.info(chars_lst[idx].shape)
        ax.imshow(chars_lst[idx], cmap='gray')
    plt.show()
