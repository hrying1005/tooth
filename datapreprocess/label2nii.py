import os
import SimpleITK as sitk
import numpy as np
from PIL import Image

src_path = "/Users/sdf/Desktop/pic/1/multi/3dimgseg/37/segmentation.nii.gz"
label_path = "/Users/sdf/Desktop/udt81/37"
save_path = "/Users/sdf/Desktop"
image = sitk.ReadImage(src_path)
image_np = sitk.GetArrayFromImage(image)
label_np = np.zeros_like(image_np)

join = os.path.join
anno_path = os.listdir(label_path)
anno_path.sort(key=lambda x: int(x.split('.png')[0]))
# anno_path.reverse()
if len(anno_path) != label_np.shape[0]:
    print("len(anno)!=label_np.shape[0]")
    exit()
i = 0
for anno in anno_path:
    label = Image.open(join(label_path, anno)).convert("L")
    label = label.resize((256, 256), Image.NEAREST)
    # label = np.flipud(label)
    label_np[i, :, :] = np.array(label).astype(np.int)
    i += 1
label_np = label_np.astype(np.int32)
# label_np[label_np < 200] = 0
# label_np[label_np > 0] = 1
# label_np[label_np == 44] = 1
# label_np[label_np == 20] = 2
# label_np[label_np == 88] = 3
# label_np[label_np == 121] = 4
label_np[label_np == 49] = 1
label_np[label_np == 22] = 2
label_np[label_np == 98] = 3
label_np[label_np == 134] = 4
label_np[label_np > 4] = 0
label = sitk.GetImageFromArray(label_np)

label.SetDirection(image.GetDirection())
label.SetOrigin(image.GetOrigin())
label.SetSpacing(image.GetSpacing())
sitk.WriteImage(label, join(save_path, "udt80.nii.gz"))
print("Save 2D label to 3D.")
