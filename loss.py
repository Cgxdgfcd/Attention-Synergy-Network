from tensorflow import nn
import tensorflow as tf
from keras import backend as K
from keras.backend import binary_crossentropy
from keras.losses import Loss, binary_crossentropy


class WBEC(Loss):
    def __init__(self, weight=2.5):
        super(WBEC, self).__init__()
        self.weight = weight

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        epslion_ = tf.constant(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epslion_, 1.0 - epslion_)

        wbce = self.weight * y_true * tf.math.log(y_pred + K.epsilon())
        wbce += (1 - y_true) * tf.math.log(1 - y_pred + K.epsilon())

        return -wbce

