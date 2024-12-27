# 计算影像组学特征
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor 

# 特征计算函数
def catch_features(imagePath, maskPath, label_num=1):
    settings = {}
    settings['binWidth'] = 25  
    settings['label'] = label_num
    settings['Interpolator'] = 3
    settings['resampledPixelSpacing'] = [1, 1, 1]
    settings['normalize'] = True
    settings['normalizeScale'] = 255
    # settings['sigma'] = [1, 3]
    # settings['wavelet'] = 'db'

    # 提取特征
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    # print('Extraction parameters:\n\t', extractor.settings)
    # extractor.enableImageTypeByName('LoG')
    # extractor.enableImageTypeByName('Wavelet')
    extractor.enableAllFeatures()

    result = extractor.execute(imagePath, maskPath)
    return result

# 计算整个参数
# dataset_list = ["xiangyang", "kits", "tongji", "henan"]
dataset_list = ["tongji"]

df = pd.read_excel("./dataset/class.xlsx")
df_select = df[(df['dataset'] == 'tongji') & (df['exclusion'] == 1)]
namelist = df_select['name'].tolist()
# 把npy改成nii.gz
namelist = [name.replace('.npy', '.nii.gz') for name in namelist]

for dataset in dataset_list:
    data_path = './dataset/new/crop_image'
    mask_path = './dataset/new/ring_5'
    save_path = './dataset/new/ring_5.csv'
    
    # 根据列名来找影像及勾画文件路径
    # pathlist = os.listdir(data_path)
    rad_df = pd.DataFrame()
    for name in namelist:
        imagePath = os.path.join(data_path, name)
        maskPath = os.path.join(mask_path, name)
        print(imagePath, maskPath)
        result = catch_features(imagePath, maskPath, label_num=1)
        rad_series = pd.DataFrame(result.values(), index=result.keys(), columns=[name.replace('.nii.gz', '')])
        rad_df = pd.concat([rad_df, rad_series], axis=1)

    rad_feature = rad_df.transpose()
    rad_feature.to_csv(save_path)