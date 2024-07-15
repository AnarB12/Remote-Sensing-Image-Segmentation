import tensorflow as tf
from keras.applications import ResNet50
from keras.layers import (Conv2D, 
                          Conv2DTranspose, 
                          Concatenate, 
                          Input, 
                          BatchNormalization, 
                          Activation)

from keras.models import Model

def unet_resnet_backbone(input_shape):
    # Encoder (ResNet)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the ResNet layers to use as a feature extractor

    # Extract the layers for skip connections
    # These layers will be used in the decoder part of U-Net for concatenation
    layer_names = [
        'input_1',        # 1024x1024
        'conv1_relu',     # 512x512
        'conv2_block3_out',  # 256x256
        'conv3_block4_out',  # 128x128
        'conv4_block6_out',  # 64x64
        'conv5_block3_out'   # 32x32
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Decoder (U-Net)
    inputs = base_model.input

    # Start decoding from the bottleneck layer
    x = layers[-1]
    for i in range(4, 0, -1):
        # Upsampling
        x = Conv2DTranspose(256 // (2 ** (5 - i)), (2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Concatenation with corresponding encoder layer
        x = Concatenate()([x, layers[i]])

        # Convolution blocks
        x = Conv2D(256 // (2 ** (5 - i)), (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(256 // (2 ** (5 - i)), (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # Final upsampling and concatenation with the initial input layer
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Concatenate()([x, layers[0]])

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Output layer
    outputs = Conv2D(6, (1, 1), activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
  input_shape = (1024, 1024, 3)
  model = unet_resnet_backbone(input_shape)
  model.summary()
