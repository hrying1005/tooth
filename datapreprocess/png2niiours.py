import os
import SimpleITK as sitk
import numpy as np
from PIL import Image


def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr


def flip90_right(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.transpose(new_arr)[::-1]
    return new_arr


src_path = "D:/3dslicer_workspace/CAOJINGWEN1_2000_6_30_nii/CAOJINGWEN1_2000_6_30_nii/CAOJINGWEN1_2000_6_30.nii.gz"
label_path = "D:/3dslicer_workspace/CAOJINGWEN1_2000_6_30"
save_path = "D:/3dslicer_workspace/CAOJINGWEN1_2000_6_30_nii"
image = sitk.ReadImage(src_path)
image_np = sitk.GetArrayFromImage(image)
image_np = np.rot90(image_np, 1)  # 旋转90度
label_np = np.zeros_like(image_np)
label_np = label_np.transpose(1, 0, 2)

join = os.path.join
anno_path = os.listdir(label_path)
anno_path.sort(key=lambda x: int(x.split('.png')[0]))
anno_path = anno_path[::-1]

if len(anno_path) != label_np.shape[0]:
    print("len(anno)!=label_np.shape[0]")
    exit()
# i = 0
i = 1  # ours

for anno in anno_path:
    label = Image.open(join(label_path, anno)).convert("L")
    label = label.transpose(Image.FLIP_TOP_BOTTOM)
    label = np.asarray(label.resize((512, 512), resample=0))
    # label = np.fliplr(label)  # ours
    # label = flip90_right(label)  # ours
    label = flip90_right(label)
    label_np[label_np.shape[0] - i, :, :] = label.astype(np.int)  # ours
    # label_np[i, :, :] = label.astype(np.int)

    i += 1

label_np = label_np.astype(np.int32)
label_np[label_np != 0] = 1
label = sitk.GetImageFromArray(label_np)

label.SetDirection(image.GetDirection())
label.SetOrigin(image.GetOrigin())
label.SetSpacing(image.GetSpacing())
sitk.WriteImage(label, join(save_path, "CAOJINGWEN1_2000_6_30.nii.gz"))
print("Save 2D label to 3D.")
