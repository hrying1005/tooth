# import numpy as np
# import matplotlib.pyplot as plt
# # cat_data = np.load('cat.npz')
# # dog_data = np.load('dog.npz')
# data = np.load('D:/3dslicer_workspace/code/R50+ViT-B_16.npz')
# print('包括：', data.files)
# print(data['Transformer/encoder_norm/bias'])
# # path="D:/3dslicer_workspace/code/R50+ViT-B_16.npz"
# # data=np.load(path)
# # x_train=data['image']*255
# # la_train=data['label']*255
# # plt.subplot(121)
# # plt.imshow(x_train)
# # plt.subplot(122)
# # plt.imshow(la_train)
# # plt.show()

import os
import numpy as np
# x = 'ok'
# print(x)

# import os
# class BatchRename():
#     '''
#     批量重命名文件夹中的图片文件
#     '''
#
#     def __init__(self):
#         self.path = 'D:\\3dslicer_workspace\\label_png\\CAOYUE2_2003_12_28'
#
#     def rename(self):
#         filelist = os.listdir(self.path)
#         i = 0
#         for item in filelist:
#             if item.endswith('.png'):
#                 str = os.listdir
#                 src = os.path.join(os.path.abspath(self.path), item)
#                 dst = os.path.join(os.path.abspath(self.path), '0_' +  + '.png')
#                 try:
#                     os.rename(src, dst)
#                     print('converting %s to %s ...' % (src, dst))
#                     i = i + 1
#                 except:
#                     continue
#         print('total %d to rename & converted %d pngs' % (total_num, i))
#
#
# if __name__ == '__main__':
#     demo = BatchRename()
#     demo.rename()


#rename_bat.py
# import os
# #suffixName = 'CL_'
# for root,dirs,files in os.walk('D:/3dslicer_workspace/label_png/CAOYUE2_2003_12_28'):
#     for FileName in files:
#         print(FileName)
#         FileRename = 'CL_' + FileName
#         os.rename(FileName, FileRename)

# import os
# import sys
#
# if __name__ == "__main__":
#
#     folder_name = "D:/3dslicer_workspace/labels"  # 获取文件夹的名字，即路径
#     file_names = os.listdir(folder_name)  # 获取文件夹内所有文件的名字
#     i = 0
#
#     for name in file_names:  # 如果某个文件名在file_names内
#         old_name = folder_name + '/' + name  # 获取旧文件的名字，注意名字要带路径名
#         new_name = folder_name + '/' + str(i)  # 定义新文件的名字，这里给每个文件名前加了前缀 a_
#         os.rename(old_name, new_name)  # 用rename()函数重命名
#         print(new_name)  # 打印新的文件名字
#         i = i+1

#iimport torch
# path = '/content/drive/MyDrive/MyTransunet/code/TransUNet/model/TU_own224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs3_224/epoch_149.pth'
# pretrained_dict = torch.load(path)
# for k, v in pretrained_dict.items():  # k 参数名 v 对应参数值
#         print(k)

# import os
# class BatchRename():
#
#     def __init__(self):
#         self.path = "D:/3dslicer_workspace/images"  # 图片的路径
#
#     def rename(self):
#         for i in range(610):
#             src = os.path.join('D:/3dslicer_workspace/temp', str(i)+ '.png')
#             dst = os.path.join('D:/3dslicer_workspace/temp', str(i+1220) + '.png')
#
#             try:
#                 os.rename(src, dst)
#                 print('converting %s to %s ...' % (src, dst))
#                 i = i + 1
#             except Exception as e:
#                 print(e)
#                 print('rename dir fail\r\n')
#
# if __name__ == '__main__':
#     demo = BatchRename()  # 创建对象
#     demo.rename()  # 调用对象的方法


# from PIL import Image
# import os
#
# # 获得文件夹下所有文件
# filePath = '/home/datadev/zf/U-2-Net-master/test_data/u2net_bce_itr_10000_train_0.018324_tar_0.001298_10more_results/'
# filenames = os.listdir(filePath)
#
# # 指定保存的文件夹
# outputPath = '/home/datadev/zf/U-2-Net-master/test_data/label/'
#
# # 迭代所有图片
# for filename in filenames:
#     # 读取图像
#     im = Image.open(filePath + filename)
#
#     # 指定逆时针旋转的角度
#     im_rotate = im.rotate(270)
#
#     # 保存图像
#     im_rotate.save(outputPath + filename)


# import os
# for file in os.listdir('D:/3dslicer_workspace/data_21-30/label/DINGYI_2000_12_19'):
#     name = file.replace(' ', '')
#     new_name = name[18:30]
#     os.rename(file, new_name)


# import os
# class BatchRename():
#
#     def __init__(self):
#         self.path = "D:/3dslicer_workspace/images"  # 图片的路径
#
#     def rename(self):
#         for i in range(64):
#             src = os.path.join('D:/3dslicer_workspace/teeth_345/data/image_png/a_temp', str(i+225) + '.png')
#             dst = os.path.join('D:/3dslicer_workspace/teeth_345/data/image_png/a_temp', str(i+1203) + '.png')
#
#             try:
#                 os.rename(src, dst)
#                 print('converting %s to %s ...' % (src, dst))
#                 i = i + 1
#             except Exception as e:
#                 print(e)
#                 print('rename dir fail\r\n')
#
# if __name__ == '__main__':
#     demo = BatchRename()  # 创建对象
#     demo.rename()  # 调用对象的方法

# import os
# import sys
#
# if __name__ == "__main__":
#
#     folder_name = "D:/3dslicer_workspace/label_png/CHENNING_2001_12_27"  # 获取文件夹的名字，即路径
#     file_names = os.listdir(folder_name)  # 获取文件夹内所有文件的名字
#
#     for name in file_names:  # 如果某个文件名在file_names内
#         old_name = folder_name + '/' + name  # 获取旧文件的名字，注意名字要带路径名
#         new_name = folder_name + '/' + 'CHENNING_2001_12_27_' + name  # 定义新文件的名字，这里给每个文件名前加了前缀 a_
#         os.rename(old_name, new_name)  # 用rename()函数重命名
#         print(new_name)  # 打印新的文件名字

#折线图
import matplotlib.pyplot as plt
import numpy as np

# epoch,acc,loss,val_acc,val_loss
x_axis_data = [1, 2, 3, 4, 5, 6, 7, 8,9,10]
y_axis_data1 = [95.57,96.85,96.65,97.70,97.02,95.60,96.73,96.40,97.65,95.97]
y_axis_data2 = [93.60,94.25,94.05,95.63,95.00,93.51,94.53,94.04,95.55,93.91]
y_axis_data3 = [90.30,90.75,91.01,91.73,91.00,89.91,91.03,90.50,92.00,90.91]
#y_axis_data3 = [82, 83, 82, 76, 84, 92, 81]

# 画图
plt.plot(x_axis_data, y_axis_data1, 'go-', alpha=0.5, linewidth=2, label='TransUnet')  # '
plt.plot(x_axis_data, y_axis_data2, 'rs-', alpha=0.5, linewidth=2, label='U-Net++')
plt.plot(x_axis_data, y_axis_data3, 'b*-', alpha=0.5, linewidth=2, label='U-Net')

plt.legend(loc="upper right")  # 显示上面的label
plt.xlabel('case')
plt.ylabel('Dice(%)')  # accuracy

# plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()

# #柱状图
# import matplotlib.pyplot as plt
#
# plt.rcParams["font.family"] = "kaiTi"
# nums = [855, 918, 1229, 1418, 1199,1450]
# x_ticks = [2016, 2017, 2018, 2019, 2020, 2021]
# x = range(0, len(nums))
# # 显示x坐标对应位置的内容，就是位置更名函数，plt.xticks()
# plt.xticks(x, x_ticks)  # rotation="vertical" 倒转九十度
# plt.bar(x, nums, width=0.5, label="市场规模")
# # plt.text(a,b,b)数据显示的横坐标、显示的位置高度、显示的数据值的大小
# for a, b in zip(x, nums):
#     plt.text(a, b + 2, b, ha='center', va='bottom')
# plt.title('2016-2021年中国口腔医疗服务市场规模趋势图')
# plt.xlabel("年份")
# plt.ylabel("规模/亿元")
# plt.legend(loc="upper left")  # 显示标签
# plt.show()




