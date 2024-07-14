"""
The SegNet architecture is based on the work by Badrinarayanan, Kendall, and Cipolla.
The following is implementation in Keras in the repository by danielenricocahall:
https://github.com/danielenricocahall/Keras-SegNet.

Credit:
- Original SegNet Paper: Badrinarayanan, V., Kendall, A., & Cipolla, R. (2015).
  "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation."
  arXiv preprint arXiv:1511.00561.
- Keras implementation: https://github.com/danielenricocahall/Keras-SegNet

Example usage:
    model = SegNet(input_shape=(256, 256, 3), n_labels=21)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
"""

import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Convolution2D, Conv2D
from keras.layers import BatchNormalization


class MaxPoolingWithArgmax2D(Layer):
    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        ksize = [1, *pool_size, 1]
        padding = padding.upper()
        strides = [1, *strides, 1]
        output, argmax = tf.nn.max_pool_with_argmax(
            inputs,
            ksize=ksize,
            strides=strides,
            padding=padding)

        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(
            self, 
            size=(2, 2), 
            **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        mask = K.cast(mask, 'int32')
        input_shape = tf.shape(updates, out_type='int32')

        if output_shape is None:
            output_shape = (
                input_shape[0],
                input_shape[1] * self.size[0],
                input_shape[2] * self.size[1],
                input_shape[3])

        ret = tf.scatter_nd(K.expand_dims(K.flatten(mask)),
                            K.flatten(updates),
                            [K.prod(output_shape)])

        input_shape = updates.shape
        out_shape = [-1,
                     input_shape[1] * self.size[0],
                     input_shape[2] * self.size[1],
                     input_shape[3]]
        return K.reshape(ret, out_shape)

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3]
        )
    


def segnet(input_shape,
           n_labels,
           num_filters=32,
           output_mode="sigmoid"):
    
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(num_filters, (3, 3), padding="same", 
                           kernel_initializer='he_normal')(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_1)

    conv_2 = Convolution2D(2 * num_filters, (3, 3), padding="same", 
                           kernel_initializer='he_normal')(pool_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_2)

    conv_3 = Convolution2D(2 * num_filters, (3, 3), padding="same", 
                           kernel_initializer='he_normal')(pool_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_3)

    conv_4 = Convolution2D(4 * num_filters, (3, 3), padding="same", 
                           kernel_initializer='he_normal')(pool_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_4)

    unpool_1 = MaxUnpooling2D()([pool_4, mask_4])

    conv_5 = Convolution2D(2 * num_filters, (3, 3), padding="same", 
                           kernel_initializer='he_normal')(unpool_1)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)

    unpool_2 = MaxUnpooling2D()([conv_5, mask_3])

    conv_6 = Convolution2D(2 * num_filters, (3, 3), padding="same", 
                           kernel_initializer='he_normal')(unpool_2)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)

    unpool_3 = MaxUnpooling2D()([conv_6, mask_2])

    conv_7 = Convolution2D(num_filters, (3, 3), padding="same", 
                           kernel_initializer='he_normal')(unpool_3)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    unpool_4 = MaxUnpooling2D()([conv_7, mask_1])

    conv_8 = Convolution2D(n_labels, (1, 1), padding="same", 
                           kernel_initializer='he_normal')(unpool_4)
    conv_8 = BatchNormalization()(conv_8)
    outputs = Activation(output_mode)(conv_8)

    model = Model(inputs=inputs, outputs=outputs)

    return model

if __name__ == "__main__":
    input_shape = (512, 512, 6)
    num_classes = 1
    model = segnet(input_shape, num_classes)
    model.summary()

