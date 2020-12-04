import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.models.train_model import visualize_history
from src.models import segmentation
from src.data.imageprocess import (pipeline_single, read_image,
                                   grayscale, threshold_image, dilate_image,
                                   blur_image, sample_image)
from src.models.segmentation import detect_contours, segment_image


# TODO visualize image processing pipeline
def visualize_image_process():
    img = read_image(sample_image)
    grayed = grayscale(img)
    threshed = threshold_image(grayed)
    dilated = dilate_image(threshed, ksize=(5, 5), iters=1)
    blurred = blur_image(dilated, ksize=(3, 3), div=15, gauss=True)
    segmented = detect_contours(blurred)[0]
    visuals_lst = [img, grayed, threshed, dilated, blurred, segmented]

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(11, 6), dpi=200)

    for graph, ax in zip(visuals_lst, ax.flatten()):
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if a.all(graph, visuals_lst[0]):
            ax.imshow(graph)
        else:
            ax.imshow(graph, cmap='gray')

        # ax[0][0].imshow(img)
        # ax[0][1].imshow(grayed, cmap='gray')
        # ax[1][0].imshow(threshed, cmap='gray')
        # ax[1][1].imshow(dilated, cmap='gray')
        # ax[2][0].imshow(blurred, cmap='gray')
        # ax[2][1].imshow(segmented, cmap='gray')
    plt.tight_layout()
    plt.show()


def visualize_segmentation(image=sample_image):
    img = pipeline_single(sample_image, dilatekernel=(3, 3),
                          blurkernel=(5, 5), div=25, gauss=True)
    chars_lst = segment_image(img)
    fig2, ax2 = plt.subplots(nrows=1, ncols=7, figsize=(11, 6), dpi=200)
    for idx, ax in enumerate(ax2.flatten()):
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if idx > len(chars_lst)-1:
            break
        ax.imshow(chars_lst[idx], cmap='gray')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize_image_process()
    visualize_segmentation()
    # model = load_model("./models/model.h5")
    # visualize_history(model)
