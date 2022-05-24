import numpy as np
import os
import imageio
from PIL import Image
import medpy.io as medio
import nibabel as nib
import matplotlib.pyplot as plt


# data_nii_pth = "/Users/sdf/Desktop/use/img"
# label_nii_pth = "/Users/sdf/Desktop/use/seg"
# save_data_pth = "/Users/sdf/Desktop/trainB"
# save_label_pth = "/Users/sdf/Desktop/trainBgt"

data_nii_pth = "D:/3dslicer_workspace/data_nii/CHENYANG_2000_1_14"
label_nii_pth = "D:/3dslicer_workspace/label_nii/CHENYANG_2000_1_14"
save_data_pth = "D:/3dslicer_workspace/data_png"
save_label_pth = "D:/3dslicer_workspace/label_png"

join = os.path.join


def readnii(path):
    img = nib.load(path)
    imgdata = img.get_fdata()
    return imgdata


def hu_cut(img, min, max):
    img = np.clip(img, min, max)
    return img


def savepng(data_nii_pth, label_nii_pth, save_data_pth, save_label_pth):
    path_data = os.listdir(data_nii_pth)

    for case in path_data:
        src_data_path = join(data_nii_pth, case)
        src_label_path = join(label_nii_pth, case)

        src_data = readnii(src_data_path)
        # src_data = hu_cut(src_data, -300, 500)  # hu值截断
        # src_data = hu_cut(src_data, 0, np.max(src_data))
        src_label = readnii(src_label_path)

        (x, y, z) = src_data.shape
        for k in range(x):
            # datax = np.zeros((256, 256), dtype=np.uint8)
            slice_data = src_data[k, :, :]
            datax = np.asarray(slice_data)
            datax = np.rot90(datax, 1)

            slice_label = src_label[k, :, :]
            datay = np.asarray(slice_label)
            datay = np.rot90(datay, 1)

            savepath_data = join(save_data_pth, '{}'.format(case.split('.')[0]))
            if not os.path.exists(savepath_data):
                os.makedirs(savepath_data)
            imageio.imwrite(join(savepath_data, '{}.png').format(k), datax)

            savepath_label = join(save_label_pth, '{}'.format(case.split('.')[0]))
            if not os.path.exists(savepath_label):
                os.makedirs(savepath_label)
            imageio.imwrite(join(savepath_label, '{}.png').format(k), datay)

        print("finish {}".format(case))


if __name__ == '__main__':
    savepng(data_nii_pth, label_nii_pth, save_data_pth, save_label_pth)
