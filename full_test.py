from src.data.imageprocess import pipeline_single
from src.models.segmentation import detect_contours, segment_image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("START")
data_directory = os.path.join(os.getcwd(), "data")
default_image_directory = os.path.join(data_directory, "external/2_recognition/license_synthetic/license-plates")
default_interim_directory = os.path.join(data_directory, "interim/2_recognition/")
default_processed_directory = os.path.join(data_directory, "processed/2_recognition/")

processed_directory = os.path.join(data_directory, "processed/2_recognition")
train_directory = os.path.join(processed_directory, "train_set")

filename_list = os.listdir(default_image_directory)
filepath_list = [os.path.join(train_directory, file) for file in filename_list]
images_array = np.zeros((480000,7,30,30))

problem_images = list()
problem_images = ['60lab03', '25la931', '14la558', '06la048', '09la479', '19la001', '39dla66', '43lar06', '25mtp33', '14bla07', '80fla76', '03lad67', '24hla35', '63lla42', '07vla57', '70la655', '23la844', '19lae73', '19lae21', '67lar77', '17ala91', '34pla75', '02mla21', '78la592', '63la123', '76lad46', '17la182', '34rla93', '12ala88', '70la565', '05la608', '06lab50', '52la479', '54zla57', '64ola39', '58las40', '55lay16', '03lay30', '04la856', '14la337', '31la978', '42bla51', '60laf70', '79la525', '32la444', '46lay01', '71la901', '21bla85', '20mla62', '68lad55', '25la929', '43kla95', '65la559', '47lam91', '58la270', '25la659', '09la725', '59la936', '52lan77', '12lag78', '73lay60', '13ola20', '60laf65']

for idx, filepath in zip(range(len(filepath_list)//1), filepath_list):
    # if filepath[-11:-4].lower().replace('-', '') in problem_images:
    problem_img = pipeline_single(filepath, dilatekernel=(3,3), blurkernel=(3,3))
    segments = segment_image(problem_img)
    try:
        images_array[idx] = segments
        if len(segments) != 7:
            break
    except:
        print(filepath)
        break


print("NEXT")

file = os.path.join(train_directory, file)
print("##############", file, "###################")
problem_img = pipeline_single(file, dilatekernel=(3,3), blurkernel=(5,5), gauss=True)
print(problem_img.shape)
# kernel = np.ones((3,3), dtype=np.uint8)
# problem_img = cv2.erode(problem_img, kernel=kernel, iterations=1)
# problem_img = cv2.threshold(problem_img, 200, 255, cv2.THRESH_BINARY)[1]
plt.imshow(problem_img, cmap='gray')
plt.show()

plt.imshow(detect_contours(problem_img)[0], cmap='gray')
plt.show()
print(detect_contours(problem_img)[1])

chars_lst = segment_image(problem_img)
fig2, ax2 = plt.subplots(nrows=3, ncols=3, figsize=(11,6), dpi=200)
for idx, ax in enumerate(ax2.flatten()):
    if idx > len(chars_lst)-1:
        break
    print(chars_lst[idx].shape)
    ax.imshow(chars_lst[idx], cmap='gray')
plt.show()


print("COMPLETE")





