from PIL import Image
import numpy as np
from keras.preprocessing import image
import os

# 生成图片地址和对应标签
file_dir1 = "./dataset_isic18/ISIC2018_Task1-2_Training_Input"
save_path1 = "./dataset_isic18/preview/img"
file_dir2 = "./dataset_isic18/ISIC2018_Task1_Training_GroundTruth"
save_path2 = "./dataset_isic18/preview/mask"

if not os.path.exists(save_path1):
    os.makedirs(save_path1)
if not os.path.exists(save_path2):
    os.makedirs(save_path2)

shape = (192, 256)
seed = 40
# 设置生成器参数
datagen = image.ImageDataGenerator(
    fill_mode='nearest',  # 操作导致图像缺失时填充方式。constant,nearest,eflect,wrap
    rotation_range=90,  # 指定旋转角度范围
    zoom_range=0.2,
    horizontal_flip=True,  # 随机对图片执行水平翻转操作
    vertical_flip=True,  # 对图片执行上下翻转操作
    data_format='channels_last')
gen1 = datagen.flow_from_directory(file_dir1,
                                   target_size=shape,
                                   batch_size=15,
                                   class_mode='input',
                                   save_to_dir=save_path1,
                                   save_prefix='tran_',
                                   seed=seed,
                                   save_format='jpg')
gen2 = datagen.flow_from_directory(file_dir2,
                                   target_size=shape,
                                   batch_size=15,
                                   class_mode='input',
                                   save_to_dir=save_path2,
                                   save_prefix='tran_',
                                   seed=seed,
                                   save_format='png')
import math

step = math.ceil(len(gen1.classes) / gen1.batch_size)
# 把数据扩充4倍
for i in range(4 * step):
    gen1.next()
    gen2.next()

