import numpy as np
import matplotlib.pyplot as plt
import  os
import pandas as pd
import SimpleITK as sitk

# 计算深度学习特征
from monai.networks.nets import resnet50 as resnet50_monai
import torch
model = resnet50_monai(pretrained=False, n_input_channels=1, widen_factor=2, conv1_t_stride=2, feed_forward=False)
model.load_state_dict(torch.load('./pretrain/fmcib.torch')['trunk_state_dict'])
device = torch.device('cuda:0')
model.to(device)
model.eval()

def resize_image(itkimage, new_size, new_spacing, resamplemethod):
    """
    resample image and label
    :param itkimage:
    :param new_size:
    :param new_spacing:
    :param resamplemethod:
    :return:
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(itkimage)   # 需要重新采样的目标图像
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    imgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return imgResampled
# 读取图像
# excel_path = './data/clinical_feature1.xlsx'
csv_path = './data/coord.csv'

data = pd.read_csv(csv_path)
name = data['name']
feature = pd.DataFrame()
for i in range(len(name)):
    image_path = data['image_path'][i]
    print(image_path)
    image = sitk.ReadImage(image_path)
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    # 重采样为1*1*1
    target_spacing = [1, 1, 1]
    scale_factor = [osp / tsp for osp, tsp in zip(original_spacing, target_spacing)]
    target_size = [int(round(osz * sf)) for osz, sf in zip(original_size, scale_factor)]
    resample_image = resize_image(image, target_size, target_spacing, resamplemethod=sitk.sitkBSpline)
    new_image = sitk.GetArrayFromImage(resample_image)
    print(new_image.shape)
    x = data['coordX'][i]
    y = data['coordY'][i]
    z = data['coordZ'][i]

    # 为50*50*50的图像
    img = new_image[x-25:x+25, y-25:y+25, z-25:z+25]
    # 查看某一层
    # image0 = img[25, :, :]
    # plt.imshow(image0, cmap='gray')
    # plt.show()
    # print(img.shape)
    new_img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
    new_img = new_img.astype(np.float32)
    print(new_img.shape)

    # 保存到cuda上
    new_img = torch.from_numpy(new_img).to(device)
    new_img = new_img.unsqueeze(0)

    # 计算特征
    with torch.no_grad():
        output = model(new_img)
        output = output.squeeze(0).cpu().numpy()
        print(output)
    feature_series = pd.Series(output, name=name[i])
    feature = pd.concat([feature, feature_series],  axis=1)
    print(feature.shape)
    # break
feature.to_csv("./data/dl_feature.csv")














# 加载.npy文件
# for file in os.listdir("./data/new_data"):
#     data = np.load("./data/new_data/" + file)
#     print(data.shape)
# data = np.load("./data/new_data/AI XIAO XIANG.npy")
# data0 = np.flip(data, axis=2)
# # plt.imshow(data[2, :, :], cmap='gray')
# plt.imshow(data0[10, :, :], cmap='gray')
# plt.show()
# print(data.shape)

# root_dir = '/home/fbz/data/wch/kidney_tumor_classification/data/new_data'
# for file in os.listdir("./data/new_data"):
#     data = np.load("./data/new_data/" + file)
# img_list = os.listdir(root_dir)
# img_list.sort()
# data_path = '/home/fbz/data/wch/kidney_tumor_classification/data/clinical_feature.xlsx'
# clinical_data = pd.read_excel(data_path)
# for idx in range(10):
#     name = img_list[idx]
#     label = clinical_data.loc[clinical_data['name'] == name.replace("_crop.npy", "")]['type']
#     print(type(label))
#     print(label+2)
# print(type(img_list))


