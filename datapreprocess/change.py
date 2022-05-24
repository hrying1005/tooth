import medpy.metric.binary as mmb
from PIL import Image
import os
import numpy as np
# 改变图片


def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr


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


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    # image_numpy = image_numpy.transpose
    image_pil = Image.fromarray(np.uint8(image_numpy))
    h, w = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


if __name__ == '__main__':
    gt_path = "/Users/sdf/Desktop/1"
    pd_path = "/Users/sdf/Desktop/2"

    id_list = os.listdir(gt_path)
    id_list.sort()
    join = os.path.join
    dice_list = []
    assd_list = []

    for id in id_list:
        gt_id_path = join(gt_path, id)
        pd_id_path = join(pd_path, id)
        gt_list = os.listdir(gt_id_path)
        pd_list = os.listdir(pd_id_path)
        gt_list.sort()
        pd_list.sort()
        for item in range(len(gt_list)):

            name = gt_list[item]
            gt_img = Image.open(join(gt_id_path, name)).convert('RGB')
            gt_img = gt_img.resize((256, 256))
            gt_np = np.array(gt_img)
            # gt_np[gt_np == 1] = 255
            # gt_np[gt_np != 0] = 255
            # gt_np = flip90_right(gt_np)
            # gt_np = flip90_left(gt_np)
            # gt_np = np.fliplr(gt_np)
            # gt_np = np.flipud(gt_np)
            # gt_np = flip180(gt_np)
            pd_img = Image.open(join(pd_id_path, name)).convert('RGB')
            pd_img = pd_img.resize((256, 256))
            pd_np = np.array(pd_img)

            # pd_np[pd_np == 1] = 255
            # pd_np[pd_np != 255] = 0

            # pd_np = flip90_right(pd_np)
            # pd_np = flip90_left(pd_np)
            # pd_np = np.flipud(pd_np)
            # pd_np = np.fliplr(pd_np)
            # pd_np = flip180(pd_np)

            # save_image(gt_np, join(gt_id_path, name))
            # save_image(pd_np, join(pd_id_path, name))
            gt_np = Image.fromarray(np.uint8(gt_np))
            pd_np = Image.fromarray(np.uint8(pd_np))
            gt_np.save(join(gt_id_path, name))
            pd_np.save(join(pd_id_path, name))

        print("%s finished" % id)
