import matplotlib.pyplot as plt
import numpy as np
from keras import models

from src.data.imageprocess import (
    SAMPLE_IMAGE,
    blur_image,
    dilate_image,
    grayscale,
    pipeline_single,
    read_image,
    threshold_image,
)
from src.models.segmentation import detect_contours, segment_image
from src.models.train_model import visualize_history


def visualize_image_process(grouped: bool = True):
    img: np.ndarray = read_image(SAMPLE_IMAGE)
    grayed = grayscale(img)
    threshed = threshold_image(grayed)
    dilated = dilate_image(threshed, ksize=(5, 5), iters=1)
    blurred = blur_image(dilated, ksize=(3, 3), div=15, gauss=True)
    segmented = detect_contours(blurred)[0]
    visuals_lst = [img, grayed, threshed, dilated, blurred, segmented]

    if grouped:
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(11, 6), dpi=200)
        for idx, ax in enumerate(ax.flatten()):
            if idx == 0:
                ax.imshow(visuals_lst[idx])
            else:
                ax.imshow(visuals_lst[idx], cmap='gray')
                ax.set_xticks([], [])
                ax.set_yticks([], [])
        plt.tight_layout()
        plt.show()
    else:
        names_lst = ['Unprocessed', 'Grayscale', 'Threshold', 'Erode', 'Blur', 'Contour Detection']
        for plot, name in zip(visuals_lst, names_lst, strict=False):
            fig, ax = plt.subplots(figsize=(8, 2), dpi=200)
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            plt.imshow(plot, cmap='gray')
            plt.title(name)
            plt.tight_layout()
            plt.show()
            # plt.savefig("./images/" + name + ".png")


def visualize_segmentation(image=SAMPLE_IMAGE):
    img = pipeline_single(SAMPLE_IMAGE, dilatekernel=(3, 3), blurkernel=(5, 5), div=25, gauss=True)
    chars_lst = segment_image(img)
    fig2, ax2 = plt.subplots(nrows=1, ncols=7, figsize=(8, 2), dpi=200)
    for idx, ax in enumerate(ax2.flatten()):
        if idx == 0:
            ax.imshow(chars_lst[idx], cmap='gray')
        elif idx > len(chars_lst) - 1:
            break
        else:
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            ax.imshow(chars_lst[idx], cmap='gray')
    plt.tight_layout()
    # plt.savefig("./images/image_segments.png")
    plt.show()


if __name__ == '__main__':
    # visualize_image_process(grouped=False)
    # visualize_segmentation()
    model = models.load_model('./models/model_full')
    visualize_history(model)
