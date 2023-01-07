import os

import keras.models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import numpy as np
import glob
from PIL import Image
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from loss import WBEC
from model import AS_Net
import pickle

height = 192
width = 256


def read_from_paths(image_path_list, mask_path_list):
    # image_path_list: 图片路径列表
    # mask_path_list: 标签路径列表
    # 返回4-D图片numpy array和4-D标签numpy array
    images = []
    masks = []
    for img_path, mask_path in zip(image_path_list, mask_path_list):
        image = Image.open(img_path).convert('RGB')
        image = np.array(image, dtype=np.float32)
        image = tf.image.adjust_gamma(image / 255., gamma=1.6)

        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.float32)
        mask = np.expand_dims(mask, -1)

        images.append(image)
        masks.append(mask / 255)
    images_array = np.array(images)
    masks_array = np.array(masks)
    return images_array, masks_array


Dataset_add = 'dataset_isic18/'
Tr_add = 'ISIC2018_Task1-2_Training_Input'
Tr_ms_add = 'ISIC2018_Task1_Training_GroundTruth'

Tr_list = glob.glob(Dataset_add + Tr_add + '/*.jpg')
Tr_ms_list = glob.glob(Dataset_add + Tr_ms_add + '/*.png')

val_data = np.load('data_val.npy').astype(dtype=np.float32)
val_mask = np.load('mask_val.npy').astype(dtype=np.float32)

val_data = tf.image.adjust_gamma(val_data / 255., gamma=1.6)
val_mask = np.expand_dims(val_mask, axis=-1)
val_mask = val_mask / 255.

batch_size = 16
nb_epoch = 10
steps_per_epoch = int(np.ceil(len(Tr_list)/batch_size))


def generator(all_image_list, all_mask_list):
    cnt = 0
    while True:
        images_array, masks_array = read_from_paths(all_image_list[cnt*batch_size:(cnt+1)*batch_size],
                                                    all_mask_list[cnt*batch_size:(cnt+1)*batch_size])
        yield images_array, masks_array

        cnt = (cnt + 1) % steps_per_epoch   # total_size/batch_size
        if cnt == 0:
            state = np.random.get_state()
            np.random.shuffle(all_image_list)
            np.random.set_state(state)
            np.random.shuffle(all_mask_list)


# Build model
model = AS_Net()
model.load_weights('./checkpoint/weights.hdf5')
model.compile(optimizer=Adam(learning_rate=1e-4, decay=1e-7), loss=WBEC(), metrics=['binary_accuracy'])

mcp_save = ModelCheckpoint('./checkpoint/weights.hdf5', save_weights_only=True)
mcp_save_best = ModelCheckpoint('./checkpoint_best/weights_best.hdf5', verbose=1, save_best_only=True, save_weights_only=True,
                                mode='min')# 0.23690

history = model.fit(x=generator(Tr_list, Tr_ms_list),
                    epochs=nb_epoch,
                    verbose=1,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=(val_data, val_mask), callbacks=[mcp_save, mcp_save_best])
print(model.optimizer.get_config())
# with open('log.txt', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)
