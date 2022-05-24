import os
import random
import shutil


data_pth1 = "/Users/sdf/Desktop/ct_dpng"
label_pth1 = "/Users/sdf/Desktop/ct_lpng"
savepath = "/Users/sdf/Desktop/ct"


def data_select(data_path, label_path, savepath):
    os.makedirs(os.path.join(savepath, 'train'))
    os.makedirs(os.path.join(savepath, 'test'))
    os.makedirs(os.path.join(savepath, 'traingt'))
    os.makedirs(os.path.join(savepath, 'testgt'))
    j = 0
    theta = 0.8
    dirs = os.listdir(data_path)
    random.shuffle(dirs)
    train_num = theta * len(dirs)
    for num in dirs:
        patient_dir = os.path.join(data_path, num)
        patient_dir2 = os.path.join(label_path, num)
        imglist = os.listdir(patient_dir)
        imglist2 = os.listdir(patient_dir2)
        for i in range(len(imglist)):
            if j < train_num:
                dst_path = os.path.join(savepath, 'train',  "%s_%s" % (num, imglist[i]))
                dst_path2 = os.path.join(savepath, 'traingt', "%s_%s" % (num, imglist[i]))
            else:
                dst_path = os.path.join(savepath, 'test', "%s_%s" % (num, imglist[i]))
                dst_path2 = os.path.join(savepath, 'testgt', "%s_%s" % (num, imglist[i]))
            shutil.copy(os.path.join(patient_dir, imglist[i]), dst_path)
            shutil.copy(os.path.join(patient_dir2, imglist2[i]), dst_path2)
        j += 1


if __name__ == '__main__':
    data_select(data_pth1, label_pth1, savepath)
