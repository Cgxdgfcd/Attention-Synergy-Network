import numpy as np
import glob
from PIL import Image
import tensorflow as tf

np.set_printoptions(threshold=np.inf)

# Parameters
height = 192
width = 256
channels = 3

############################################################# Prepare ISIC 2018 data set #################################################
Dataset_add = 'dataset_isic18/'
Tr_add = 'ISIC2018_Task1-2_Training_Input'
Tr_ms_add = 'ISIC2018_Task1_Training_GroundTruth'

Va_add = 'ISIC2018_Task1-2_Validation_Input'
Va_ms_add = 'ISIC2018_Task1_Validation_GroundTruth'

Tr_list = glob.glob(Dataset_add + Tr_add + '/*.jpg')
Tr_ms_list = glob.glob(Dataset_add + Tr_ms_add + '/*.png')
Va_list = glob.glob(Dataset_add + Va_add + '/*.jpg')
Va_ms_list = glob.glob(Dataset_add + Va_ms_add + '/*.png')
print(len(Tr_list), len(Tr_ms_list), len(Va_list), len(Va_ms_list))

Data_train_2018 = np.zeros([len(Tr_list), height, width, channels])
Label_train_2018 = np.zeros([len(Tr_ms_list), height, width])
Data_validate_2018 = np.zeros([len(Va_list), height, width, channels])
Label_validate_2018 = np.zeros([len(Va_ms_list), height, width])

# for idx in range(len(Tr_list)):
#     print(idx + 1)
#     img = Image.open(Tr_list[idx]).convert('RGB')
#     img = img.resize((width, height))
#     img.save(Tr_list[idx])
#     # Data_train_2018[idx, :, :, :] = img
#
#     img2 = Image.open(Tr_ms_list[idx])
#     img2 = img2.resize((width, height))
#     img2.save(Tr_ms_list[idx])
#     # Label_train_2018[idx, :, :] = img2

for idx in range(len(Va_list)):
    print(idx + 1)
    img = Image.open(Va_list[idx]).convert('RGB')
    img = np.array(img.resize((width, height))).astype(dtype=np.float32)
    Data_validate_2018[idx, :, :, :] = img

    img2 = Image.open(Va_ms_list[idx])
    img2 = np.array(img2.resize((width, height))).astype(dtype=np.float32)
    Label_validate_2018[idx, :, :] = img2

print('Reading ISIC 2018 finished')

################################################################ Make the train and test sets ########################################
# # np.save('data_train', Train_img)
# # np.save('data_test', Test_img)
np.save('data_val', Data_validate_2018)
#
# # np.save('mask_train', Train_mask)
# # np.save('mask_test', Test_mask)
np.save('mask_val', Label_validate_2018)
