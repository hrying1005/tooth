import os.path
from PIL import Image
import random
import torchvision.transforms as T
# import elasticdeform.torch as etorch
import elasticdeform as edf
import numpy as np
from tqdm import tqdm
import albumentations as A

# 数据增强
# num 控制图片数量 *num
# 可以改动edf.deform_random_grid的sigma

num_epoches = 3
root_path = "/Users/sdf/Desktop/train"
save_path = "/Users/sdf/Desktop/train_aug"

if not os.path.exists(os.path.join(save_path, "img")):
    os.makedirs(os.path.join(save_path, "img"))
    os.makedirs(os.path.join(save_path, "seg"))


def data_aug_offline(root_path, num_epoches, save_path=None):
    img_resize = T.Resize(size=256, interpolation=Image.BICUBIC)
    seg_resize = T.Resize(size=256, interpolation=Image.NEAREST)
    a_rbc = A.RandomBrightnessContrast(always_apply=False, p=0.9)  # 亮度对比度调整
    a_g = A.GaussianBlur(blur_limit=1, always_apply=False, p=0.1)

    join = os.path.join
    id_list = os.listdir(root_path)
    id_list.sort()
    img_pathes_list = []

    id_path = join(root_path, 'img')
    img_list = os.listdir(id_path)
    img_list.sort()
    img_pathes_list.extend([join(id_path, img) for img in img_list])

    for epoch in tqdm(range(num_epoches)):
        print("==============================================")
        print("Current epoch is ", epoch + 1)
        print("==============================================")
        temp_list = img_pathes_list.copy()
        # add_list = [random.sample(temp_list, 1)[0] for _ in range(add)]
        # temp_list.extend(add_list)
        for img_path in tqdm(temp_list):
            seg_path = img_path.replace('/img/', '/seg/')
            img = Image.open(img_path).convert('L')
            seg = Image.open(seg_path).convert('L')

            img_aug_dir = join(save_path, img_path.split('/')[-2])
            if not os.path.exists(img_aug_dir):
                os.makedirs(img_aug_dir)
            seg_aug_dir = img_aug_dir.replace('/img/', '/seg/')
            if not os.path.exists(seg_aug_dir):
                os.makedirs(seg_aug_dir)
            img_aug_name = str(epoch) + '_' + img_path.split('/')[-1]

            img_aug_path = join(img_aug_dir, img_aug_name)
            seg_aug_path = img_aug_path.replace('/img/', '/seg/')

            # aug_img = adjust_sharp(img_resize(img))
            aug_img = img_resize(img).convert('L')
            aug_seg = seg_resize(seg).convert('L')
            aug_img = a_rbc(image=np.array(aug_img))['image']
            aug_img = a_g(image=np.array(aug_img))['image']
            aug_img = Image.fromarray(aug_img).convert('L')

            if random.randint(0, 10) >= 3:
                [aug_img, aug_seg] = edf.deform_random_grid([np.array(aug_img), np.array(aug_seg)],
                                                            points=2, order=0, mode='nearest')

                aug_img = Image.fromarray(aug_img).convert('L')
                aug_seg = Image.fromarray(aug_seg).convert('L')
            aug_img.save(img_aug_path)
            aug_seg.save(seg_aug_path)


if __name__ == '__main__':
    data_aug_offline(root_path, num_epoches, save_path)
