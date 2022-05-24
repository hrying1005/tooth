import os
import SimpleITK as sitk
import shutil
# src = "/Users/sdf/Desktop/shuhou"
# dst = "/Users/sdf/Desktop/1"

src = "D:/3dslicer_workspace/DONGYICHEN_2009_9_8"
dst = "D:/3dslicer_workspace/data_nii"

exists = os.path.exists
join = os.path.join

def change(src_path, dst_path):
    src_case_names = [name for name in os.listdir(src_path)]
    # dst_case_names = [name for name in os.listdir(dst_path)]

    # 检查dst_path 目录下，子文件中imaging.nii.gz是否存在
    #
    # for case in os.listdir(dst_path):
    #     if not exists(join(dst_path, case, "imaging.nii.gz")):
    #         print("{0} not exists...".format(join(dst_path, case, "imaging.nii.gz")))
    #         shutil.rmtree(join(dst_path, case))
    #
    # # 如果src_path 子文件中dicom格式已经转换称.nii.gz格式，则从src_case_names中删除
    # for src_case in src_case_names:
    #     if src_case in dst_case_names:
    #         src_case_names.remove(src_case)
    #
    # for src_case in src_case_names:
    #     if exists(join(dst_path, src_case, "imaging.nii.gz")):
    #         src_case_names.remove(src_case)

    print(len(src_case_names))
    reader = sitk.ImageSeriesReader()

    for src_case in src_case_names:
        path = os.path.join(src_path, src_case)
        seriresIDs = reader.GetGDCMSeriesIDs(path)  # 根据文件夹获取序列ID
        dcm_series = reader.GetGDCMSeriesFileNames(path, seriresIDs[0])  # 选取其中一个序列ID,获得该序列的若干文件名
        reader.SetFileNames(dcm_series)  # 设置文件名
        image = reader.Execute()  # 读取dicom序列
        if not exists(join(dst_path, src_case)):
            os.mkdir(join(dst_path, src_case))
        sitk.WriteImage(image, join(dst_path, src_case, '{0}.nii.gz'.format(src_case)))
        print(src_case, "is converted..")


if __name__ == '__main__':
    change(src, dst)
