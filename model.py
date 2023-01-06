import keras.backend
import numpy as np
from keras import Model, Input
from keras.applications import VGG16
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Activation, UpSampling2D, concatenate, Multiply, GlobalAveragePooling2D, Dense, Reshape
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras import backend as K


def AS_Net(input_size=(192, 256, 3)):
    inputs = Input(input_size)
    VGGnet = VGG16(weights='imagenet', include_top=False, input_shape=(192, 256, 3))
    output1 = Model(inputs=VGGnet.inputs, outputs=VGGnet.get_layer(index=2).output)(inputs)
    output2 = Model(inputs=VGGnet.inputs, outputs=VGGnet.get_layer(index=5).output)(inputs)
    output3 = Model(inputs=VGGnet.inputs, outputs=VGGnet.get_layer(index=9).output)(inputs)
    output4 = Model(inputs=VGGnet.inputs, outputs=VGGnet.get_layer(index=13).output)(inputs)
    output5 = Model(inputs=VGGnet.inputs, outputs=VGGnet.get_layer(index=17).output)(inputs)

    merge1 = concatenate([output4, UpSampling2D((2, 2), interpolation='bilinear')(output5)], axis=-1)
    SAM1 = SAM(filters=1024)(merge1)
    CAM1 = CAM(filters=1024)(merge1)

    merge21 = concatenate([output3, UpSampling2D((2, 2), interpolation='bilinear')(SAM1)], axis=-1)
    merge22 = concatenate([output3, UpSampling2D((2, 2), interpolation='bilinear')(CAM1)], axis=-1)
    SAM2 = SAM(filters=512)(merge21)
    CAM2 = CAM(filters=512)(merge22)

    merge31 = concatenate([output2, UpSampling2D((2, 2), interpolation='bilinear')(SAM2)], axis=-1)
    merge32 = concatenate([output2, UpSampling2D((2, 2), interpolation='bilinear')(CAM2)], axis=-1)
    SAM3 = SAM(filters=256)(merge31)
    CAM3 = CAM(filters=256)(merge32)

    merge41 = concatenate([output1, UpSampling2D((2, 2), interpolation='bilinear')(SAM3)], axis=-1)
    merge42 = concatenate([output1, UpSampling2D((2, 2), interpolation='bilinear')(CAM3)], axis=-1)
    SAM4 = SAM(filters=128)(merge41)
    CAM4 = CAM(filters=128)(merge42)

    output = Synergy()((SAM4, CAM4))
    output = Activation('sigmoid')(output)

    model = Model(inputs=inputs, outputs=output)

    return model


class SAM(Model):
    def __init__(self, filters=1024):
        super(SAM, self).__init__()
        self.filters = filters

        self.conv1 = Conv2D(self.filters//4, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(self.filters//4, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv3 = Conv2D(self.filters//4, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.conv4 = Conv2D(self.filters//4, 1, activation='relu', kernel_initializer='he_normal')

        self.pool1 = MaxPooling2D((2, 2))
        self.upsample1 = UpSampling2D((2, 2), interpolation='bilinear')
        self.W1 = Conv2D(self.filters//4, 1, activation='sigmoid', kernel_initializer='he_normal')
        self.pool2 = MaxPooling2D((4, 4))
        self.upsample2 = UpSampling2D((4, 4), interpolation='bilinear')
        self.W2 = Conv2D(self.filters//4, 1, activation='sigmoid', kernel_initializer='he_normal')

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs)))
        out2 = self.conv4(inputs)

        merge1 = self.W1(Activation('relu')(self.upsample1(self.pool1(out2))))
        merge2 = self.W2(Activation('relu')(self.upsample2(self.pool2(out2))))

        out3 = merge1 + merge2

        y = Multiply()([out1, out3]) + out2

        return y


class CAM(Model):
    def __init__(self, filters, reduction_radio=16):
        super(CAM, self).__init__()
        self.filters = filters

        self.conv1 = Conv2D(self.filters//4, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(self.filters//4, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv3 = Conv2D(self.filters//4, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.conv4 = Conv2D(self.filters//4, 1, activation='relu', kernel_initializer='he_normal')

        self.gpool = GlobalAveragePooling2D()
        self.fc1 = Dense(self.filters//(4*reduction_radio), activation='relu', use_bias=False)
        self.fc2 = Dense(self.filters//4, activation='sigmoid', use_bias=False)
        self.reshape = Reshape((1, 1, self.filters))

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs)))
        out2 = self.conv4(inputs)
        out3 = self.fc2(self.fc1(self.gpool(out2)))

        y = Multiply()([out1, out3]) + out2

        return y


class Synergy(Model):
    def __init__(self, alpha=0.5, belta=0.5):
        super(Synergy, self).__init__()
        self.altha = alpha
        self.belta = belta
        self.conv = Conv2D(1, 3, padding='same', kernel_initializer='he_normal')
        self.bn = BatchNormalization()

    def call(self, inputs):
        x, y = inputs
        inputs = self.altha*x + self.belta*y
        y = self.bn(self.conv(inputs))

        return y


if __name__ == '__main__':
    # inputs = np.random.randn(16, 192, 256, 3)
    # model = AS_Net()
    # y = model(inputs)
    print(keras.backend.epsilon())


