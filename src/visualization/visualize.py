import matplotlib.pyplot as plt

#TODO visualize image processing pipeline

#TODO visualize segmentation result
chars_lst = segmentation(filename, dilatekernel=(3,3),blurkernel=(5,5), div=25, gauss=True)
    fig2, ax2 = plt.subplots(nrows=3, ncols=3, figsize=(11,6), dpi=200)
    for idx, ax in enumerate(ax2.flatten()):
        if idx > len(chars_lst)-1:
            break
        print(chars_lst[idx].shape)
        ax.imshow(chars_lst[idx], cmap='gray')
    # plt.show()

#TODO evaluation stuff
