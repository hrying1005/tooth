import medpy.metric.binary as mmb
from PIL import Image
import os
import numpy as np
if __name__=='__main__':
    gt_path = "/Users/sdf/Desktop/pic/2/multi/onlyorgan/seg"
    pd_path = "/Users/sdf/Desktop/dug"

    id_list = os.listdir(gt_path)
    id_list.sort()
    join = os.path.join
    dice_list = []
    assd_list = []
    cls_num = 4
    dice_mean = 0
    for id in id_list:
        gt_id_path = join(gt_path, id)
        pd_id_path = join(pd_path, id)
        gt_list = os.listdir(gt_id_path)
        pd_list = os.listdir(pd_id_path)
        gt_list.sort()
        pd_list.sort()

        assert len(pd_list) == len(gt_list)
        tmp_img_path = join(gt_id_path, gt_list[0])
        tmp_img = Image.open(tmp_img_path).convert('L')
        size = tmp_img.size

        gt_np = np.zeros((size[0],size[1],len(gt_list)))
        pd_np = np.zeros((size[0],size[1],len(gt_list)))
        for item in range(len(gt_list)):
            name = gt_list[item]
            gt_img = Image.open(join(gt_id_path,name)).convert('L')
            gt_np[:,:,item] = np.array(gt_img)
            pd_img = Image.open(join(pd_id_path,name)).convert('L')
            pd_img = pd_img.resize(size,Image.NEAREST)
            pd_np[:,:,item] = np.array(pd_img)

        pd_np[pd_np == 134] = 4
        pd_np[pd_np == 98] = 3
        pd_np[pd_np == 49] = 1
        pd_np[pd_np == 22] = 2
        gt_np[gt_np == 134] = 4
        gt_np[gt_np == 98] = 3
        gt_np[gt_np == 49] = 1
        gt_np[gt_np == 22] = 2

        for c in range(1, cls_num+1):
            tmp_gt = gt_np.copy()
            tmp_gt[tmp_gt!=c] = 0

            tmp_pd = pd_np.copy()
            tmp_pd[tmp_pd!=c] = 0
            dice = mmb.dc(tmp_pd, tmp_gt)
            assd = mmb.assd(tmp_pd, tmp_gt)

            # print("case {0},class:{1},dice:{2}".format(id, c,dice))
            print("case {0},class:{1},dice:{2},assd:{3}".format(id, c,dice, assd))
            dice_list.append(dice)
            assd_list.append(assd)

        print("========================")

    dice_arry = 100 * np.reshape(dice_list, [cls_num,-1]).transpose()
    assd_arry = np.reshape(assd_list, [cls_num,-1]).transpose()

    dice_mean = np.mean(dice_arry, axis=1)
    dice_std = np.std(dice_arry, axis=1)
    print('Dice:')
    print('Spleen:%.1f(%.1f)' %(dice_mean[0],dice_std[0]))
    print('Left Kidney:%.1f(%.1f)' %(dice_mean[1],dice_std[1]))
    print('Right Kidney:%.1f(%.1f)' %(dice_mean[2],dice_std[2]))
    print('Liver :%.1f(%.1f)' %(dice_mean[3],dice_std[3]))
    mean_dice = dice_mean[0] + dice_mean[1] + dice_mean[2] + dice_mean[3]
    mean_dice = mean_dice / 4.0
    print("All organs mean dice :%.4f" % mean_dice)

    assd_mean = np.mean(assd_arry,axis=1)
    assd_std = np.std(assd_arry,axis=1)
    print('ASSD:')
    print('Classs 1:%.1f(%.1f)' %(assd_mean[0],assd_std[0]))
    print('Classs 2:%.1f(%.1f)' %(assd_mean[1],assd_std[1]))
    print('Classs 3:%.1f(%.1f)' %(assd_mean[2],assd_std[2]))
    print('Classs 4:%.1f(%.1f)' %(assd_mean[3],assd_std[3]))
    mean_assd = assd_mean[0] + assd_mean[1] + assd_mean[2] + assd_mean[3]
    mean_assd = mean_assd / 4.0
    print("All organs mean assd :%.4f" % mean_assd)