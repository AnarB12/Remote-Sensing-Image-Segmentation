from keras.layers import (
    Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, 
    Concatenate, BatchNormalization, GlobalAveragePooling2D, Add, 
    Maximum, Dense, Reshape, Multiply, GlobalMaxPooling2D, Activation
)

# ======================== U-Net(s) building blocks ========================

def conv_block(x, filter_size, num_filters, dropout, num_conv_stack=2, batch_norm=False,
               kernel_initializer='he_normal', use_se=False, ratio=8, dilation_rate=1,
               SE_pooling=None, SE_aggregation=None):
    """
    Constructs a convolutional block consisting of multiple convolutional layers,
    with options for batch normalization, dropout, and Squeeze-and-Excitation 
    (SE) block integration.

    Parameters:
    - x: Input tensor to the convolutional block.
    - filter_size: Integer, size of the convolution kernel (filter). 
      It defines the height and width of the 2D convolution window.
    - num_filters: Integer, the number of output filters in the convolution.
    - dropout: Float between 0 and 1. Fraction of the input units to drop. 
      If 0, no dropout is applied.
    - num_conv_stack: Integer, the number of convolutional layers in the block. 
      Default is 2.
    - batch_norm: Boolean, whether to include batch normalization after 
      each convolution layer. Batch normalization can improve the learning 
      dynamics and sometimes performance.
    - kernel_initializer: String or keras.initializers.Initializer, initializer 
      for the kernel weights matrix.
    - use_se: Boolean, whether to include a Squeeze-and-Excitation block after 
      the convolution layers.
    - ratio: Integer, reduction ratio for the SE block. Only used if use_se is 
      True. It determines the bottleneck structure in the SE block for channel 
      recalibration.
    - dilation_rate: Integer, tuple of two integers, or a list of integers, 
      specifying the dilation rate to use for atrous convolution. If a single 
      integer, the same value is used for all layers. If a list, each element 
      is applied to the corresponding convolution layer in the block.
    - SE_pooling: The pooling method used in the SE block, if any.
    - SE_aggregation: The aggregation method used in the SE block, if any.

    Returns:
    - The output tensor after applying the convolutional block.
    """
    for i in range(num_conv_stack):
        is_list = isinstance(dilation_rate, list)
        current_dilation_rate = dilation_rate[i] if is_list else dilation_rate

        x = Conv2D(num_filters, (filter_size, filter_size), padding='same', 
                   kernel_initializer=kernel_initializer, 
                   dilation_rate=current_dilation_rate)(x)
        if batch_norm:
            x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

    if use_se:
        x = scSE_block(x, ratio=ratio, spatial_squeeze=True, 
                       channel_squeeze=True, pooling=SE_pooling, 
                       aggregation=SE_aggregation)

    if dropout > 0:
        x = Dropout(dropout)(x)

    return x


def res_conv_block(x, filter_size, num_filters, dropout, num_conv_stack=2, batch_norm=False,
                   kernel_initializer='he_normal', use_se=False, ratio=8, dilation_rate=1,
                   SE_pooling=None, SE_aggregation=None):
    """
    Constructs a residual convolutional block that includes multiple convolutional 
    layers and a shortcut connection. This block can optionally incorporate a 
    Squeeze-and-Excitation (SE) block.

    The parameters for the convolutional layers, batch normalization, dropout, 
    and SE block are similar to those in the `conv_block` function. Refer 
    to `conv_block` for parameter details.

    The block establishes a residual path by adding the output of a shortcut 
    convolutional layer (with kernel size 1x1) to the output of the main 
    convolutional layers. This approach helps mitigate the vanishing gradient 
    problem and allows the learning of identity functions, contributing to 
    deeper network training.

    Parameters:
    - x: Input tensor to the residual convolutional block.
    - filter_size, size, dropout, batch_norm, kernel_initializer, 
      kernel_regularizer, use_se, ratio, dilation_rate: Refer to `conv_block` 
      for a detailed explanation of these parameters.
    """
    activation_last = True

    # shortcut = x
    shortcut = Conv2D(num_filters, kernel_size=(1, 1), padding='same', 
                      kernel_initializer=kernel_initializer, 
                      dilation_rate=dilation_rate)(x)
    if batch_norm:
        shortcut = BatchNormalization(axis=-1)(shortcut)

    for i in range(num_conv_stack):
        is_list = isinstance(dilation_rate, list)
        current_dilation_rate = dilation_rate[i] if is_list else dilation_rate

        x = Conv2D(num_filters, (filter_size, filter_size), padding='same', 
                   kernel_initializer=kernel_initializer, 
                   dilation_rate=current_dilation_rate)(x)
        if batch_norm:
            x = BatchNormalization(axis=-1)(x)
        if activation_last:
            if i < (num_conv_stack-1):
                x = Activation('relu')(x)
        else:
            x = Activation('relu')(x)

    if dropout > 0:
        x = Dropout(dropout)(x)

    res_path = Add()([shortcut, x])

    if use_se:
        res_path = scSE_block(res_path, ratio=ratio, spatial_squeeze=True, 
                              channel_squeeze=True, pooling=SE_pooling, 
                              aggregation=SE_aggregation)

    res_path = Activation('relu')(res_path)

    return res_path


def ASPP(x, filter_size, atrous_rates, batch_norm=False, kernel_initializer='he_normal'):
    """
    Atrous Spatial Pyramid Pooling (ASPP) Module.

    Parameters:
    - x: Input tensor to the ASPP module.
    - filter_size: Integer, the number of output filters in the convolutions.
    - atrous_rates: List of integers, specifying the dilation rates for atrous 
      convolutions.
    - batch_norm: Boolean, whether to apply batch normalization after convolutions.
    - kernel_initializer: String, initializer for the convolution kernels.

    The ASPP module applies parallel atrous convolutions with different dilation 
    rates to capture multi-scale information, followed by a global average pooling 
    branch. The outputs of all branches are concatenated to form the final output.
    """
    GlobalPool = True
    num_filters = x.shape[-1]

    # 1x1 Convolution Branch
    conv_1x1 = Conv2D(num_filters, (1, 1), padding="same", 
                      kernel_initializer=kernel_initializer, use_bias=False)(x)
    if batch_norm:
        conv_1x1 = BatchNormalization(axis=-1)(conv_1x1)
    conv_1x1 = Activation('relu')(conv_1x1)

    # Atrous Convolution Branches
    atrous_branches = [conv_1x1]
    for rate in atrous_rates:
        atrous_conv = Conv2D(num_filters, (filter_size, filter_size), 
                             padding="same", kernel_initializer=kernel_initializer,
                             dilation_rate=rate, use_bias=False)(x)
        if batch_norm:
            atrous_conv = BatchNormalization(axis=-1)(atrous_conv)
        atrous_conv = Activation('relu')(atrous_conv)
        atrous_branches.append(atrous_conv)

    # Global Average Pooling Branch
    if GlobalPool:
      shape = int(x.shape[1])
      global_pool = GlobalAveragePooling2D()(x)
      global_pool = Reshape((1, 1, num_filters))(global_pool)
      global_pool = Conv2D(num_filters, (1, 1), padding='same', 
                          kernel_initializer=kernel_initializer,
                          use_bias=False)(global_pool)
      if batch_norm:
          global_pool = BatchNormalization(axis=-1)(global_pool)
      global_pool = Activation('relu')(global_pool)
      global_pool = UpSampling2D(size=(shape, shape))(global_pool)
      atrous_branches = atrous_branches + [global_pool]
      
    # Concatenate all branches
    aspp_output = Concatenate(axis=-1)(atrous_branches)  
    aspp_output = Conv2D(num_filters, (1, 1), padding='same', 
                         kernel_initializer=kernel_initializer)(aspp_output)
    if batch_norm:
      aspp_output = BatchNormalization(axis=-1)(aspp_output)
    aspp_output = Activation('relu')(aspp_output)

    return aspp_output


def scSE_block(x, ratio=8, spatial_squeeze=True, channel_squeeze=True, 
               pooling='avg', aggregation='max'):
    """
    Squeeze and Excitation Block with optional Spatial and Channel-wise 
    squeeze features. If both spatial and channel squeeze are used, an 
    element-wise max-out operation is performed on the output.

    Parameters:
    - spatial_squeeze: Enable/disable spatial squeeze.
    - channel_squeeze: Enable/disable channel squeeze.
    - pooling: Can be 'average', 'max', or 'both'. If 'both', values are added 
      together before sigmoid activation.
    - aggregation: Determines the type of aggregation method. Can be 'max', 
      'add', 'multiply', 'concatenate'.
    """
    channels = x.shape[-1]
    output_spatial = None
    output_channel = None

    if spatial_squeeze:
        fc2_pool_combined = None

        if pooling in ['avg', 'both']:
            # Spatial Squeeze - Average Pooling
            avg_pool = GlobalAveragePooling2D()(x)
            fc1_avg_pool = Dense(channels // ratio, use_bias=False)(avg_pool)
            fc1_avg_pool_activated = Activation('relu')(fc1_avg_pool)
            fc2_avg_pool = Dense(channels, use_bias=False)(fc1_avg_pool_activated)
            fc2_pool_combined = fc2_avg_pool

        if pooling in ['max', 'both']:
            # Spatial Squeeze - Max Pooling
            max_pool = GlobalMaxPooling2D()(x)
            fc1_max_pool = Dense(channels // ratio, use_bias=False)(max_pool)
            fc1_max_pool_activated = Activation('relu')(fc1_max_pool)
            fc2_max_pool = Dense(channels, use_bias=False)(fc1_max_pool_activated)
            if fc2_pool_combined is not None:
                fc2_pool_combined = Add()([fc2_pool_combined, fc2_max_pool])
            else:
                fc2_pool_combined = fc2_max_pool

        activated_combined = Activation('sigmoid')(fc2_pool_combined)
        reshaped_combined = Reshape((1, 1, channels))(activated_combined)
        output_spatial = Multiply()([x, reshaped_combined])

    if channel_squeeze:
        # Channel Squeeze
        conv = Conv2D(1, (1, 1), kernel_initializer='he_normal', use_bias=False)(x)
        conv_activated = Activation('sigmoid')(conv)
        output_channel = Multiply()([x, conv_activated])

    # Choose aggregation method based on the provided string
    if spatial_squeeze and channel_squeeze:
        if aggregation == 'max':
            output = Maximum()([output_spatial, output_channel])
        elif aggregation == 'add':
            output = Add()([output_spatial, output_channel])
        elif aggregation == 'multiply':
            output = Multiply()([output_spatial, output_channel])
        elif aggregation == 'concatenate':
            output = Concatenate(axis=-1)([output_spatial, output_channel])
    elif spatial_squeeze:
        output = output_spatial
    elif channel_squeeze:
        output = output_channel
    else:
        output = x 

    return output


def downsampling_block(x, downsample_type, pool_size, filters, filter_size, 
                       batch_norm, kernel_initializer):
    """
    Applies downsampling to the input tensor using either max pooling or convolution.

    Parameters:
    - x: The input tensor to downsample.
    - downsample_type: A string specifying the type of downsampling to apply 
                      ('maxpool' or 'conv').
    - pool_size: The size of the pool (width, height) for max pooling or stride for 
                 convolutional downsampling.
    - filters: The number of filters for convolutional downsampling. Not used for max pooling.
    - filter_size: The size of the convolution filter for convolutional downsampling.
    - batch_norm: Boolean indicating whether to use batch normalization after convolution.
    - kernel_initializer: The initialization method for kernel weights in convolutional downsampling.

    Returns:
    - The downsampled tensor.
    """
    if downsample_type == 'maxpool':
        x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)
    elif downsample_type == 'conv':
        x = Conv2D(filters, (filter_size, filter_size), strides=(pool_size, pool_size), 
                   kernel_initializer=kernel_initializer, padding='same')(x)
        if batch_norm:
            x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
    else:
        raise ValueError(f"Unsupported downsample type: '{downsample_type}'.\
                           Supported types are: ['maxpool', 'conv']")

    return x


def gated_attention(x, g, inter_channel, kernel_initializer='he_normal', 
                    batch_norm=False):
    """
    Constructs a gated attention block, combining features from two input 
    tensors through an attention mechanism. This block also includes batch 
    normalization on the output.

    Parameters:
    - x: Input tensor to the attention block.
    - g: A second input tensor that provides gating signals.
    - inter_channel: Number of filters in the intermediate layers.
    - kernel_initializer: Initializer for the kernel weights.
    - batch_norm: Boolean, whether to apply batch normalization after each 
      convolution operation.

    The attention mechanism allows the network to focus on specific features 
    in the input tensor x, guided by the gating signal. This can enhance the 
    representational capacity of the model, particularly in tasks like image 
    segmentation where focusing on relevant regions is important.
    """
    theta_x = Conv2D(inter_channel, (1, 1), padding='same', 
                     kernel_initializer=kernel_initializer)(x)
    if batch_norm:
        theta_x = BatchNormalization(axis=-1)(theta_x)

    phi_g = Conv2D(inter_channel, (1, 1), padding='same', 
                   kernel_initializer=kernel_initializer)(g)
    if batch_norm:
        phi_g = BatchNormalization(axis=-1)(phi_g)

    add_xg = Add()([phi_g, theta_x])
    act_xg = Activation('relu')(add_xg)

    psi = Conv2D(1, (1, 1), padding='same', 
                 kernel_initializer=kernel_initializer)(act_xg)
    if batch_norm:
        psi = BatchNormalization(axis=-1)(psi)
    
    sigmoid_psi = Activation('sigmoid')(psi)
    output = Multiply()([x, sigmoid_psi])

    return output


def bottleneck_block(x, conv_block_type, bottleneck_type, filters, filter_size, dropout_rate, 
                     num_conv_stack, batch_norm, kernel_initializer, atrous_rates):
    """
    Applies a bottleneck layer to the U-Net architecture.
    
    Parameters:
    - x: The input tensor to the bottleneck layer.
    - bottleneck_type: A string indicating the type of bottleneck ('conv', 'aspp', or 'conv+aspp').
    - filters: The number of filters to use in the bottleneck convolution layer.
    - filter_size: The size of the convolution filter.
    - dropout_rate: The dropout rate to apply after the convolution.
    - num_conv_stack: The number of convolution layers to stack in the bottleneck (if applicable).
    - batch_norm: Boolean indicating whether to use batch normalization.
    - kernel_initializer: The initialization method for kernel weights.
    - atrous_rates: A list of atrous rates for the ASPP module.
    
    Returns:
    - The output tensor after applying the bottleneck layers.
    """
    # Initial convolution layer in the bottleneck
    x = conv_block_type(x, filter_size, filters, dropout_rate, num_conv_stack, 
                        batch_norm, kernel_initializer, False, None, 
                        1, None, None)

    # If ASPP is part of the bottleneck, apply it after the initial convolution
    if bottleneck_type == 'aspp' or bottleneck_type == 'conv+aspp':
        atrous_rates = atrous_rates or [1, 2, 4, 6]  # Default atrous rates if not specified
        x = ASPP(x, filter_size, atrous_rates, batch_norm, kernel_initializer)

    # For 'conv+aspp', the function already adds ASPP after the convolution, 
    # no additional steps needed.
    # Note: If the bottleneck_type is 'conv', only the convolution is applied and 
    # ASPP is not included, which is handled by the initial convolution layer addition.

    return x


def Upsample(x, up_samp_size):
    """
    Defines an upsampling deconvolutional block using UpSampling2D.
    """
    x = UpSampling2D(size=(up_samp_size, up_samp_size),  
                     data_format='channels_last',
                     interpolation = "bilinear")(x)
    return x


def TransposedConv(x, filters, kernel_size, batch_norm=False, 
                   kernel_initializer='he_normal'):
    """
    Defines an upsampling deconvolutional block using Conv2DTranspose,
    followed by Batch Normalization and ReLU activation.
    """
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, 
                        strides=(2, 2), padding='same', 
                        kernel_initializer=kernel_initializer)(x)

    if batch_norm:
      x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    return x


def _upsampling_layer(x, filters, up_samp_size, upsample_type='bilinear', 
                      batch_norm=False, kernel_initializer='he_normal'):
    """
    Helper function for upsampling: either UpSampling2D or TransposedConv.
    """
    if upsample_type == 'deconv':
        return TransposedConv(x, filters, up_samp_size, batch_norm, kernel_initializer)
    elif upsample_type == 'bilinear':
        return Upsample(x, up_samp_size)
    else:
        raise ValueError(f"Unsupported upsample type: '{upsample_type}'.\
                         Supported types are: ['bilinear', 'deconv']")


def upsample_unet(x, conv_layers, i, filter_list, filter_size, dropout_rate, 
                  num_conv_stack, batch_norm, kernel_initializer, SE_block, 
                  up_samp_size, upsample_type, ratio, dilation_rate, 
                  SE_pooling, SE_aggregation):
    """
    Constructs an upsampling block for the U-Net architecture. This block 
    is responsible for upsampling the feature maps and concatenating them 
    with the corresponding feature maps from the downsampling path (skip connections).
    For convolutional parameters (filter_size, dropout_rate, num_layers, batch_norm, 
    kernel_initializer, SE_block, ratio, dilation_rate, SE_pooling, SE_aggregation), 
    refer to the documentation of the 'conv_block' function.

    Parameters:
    - x: Input tensor to the upsampling block.
    - conv_layers: List of convolutional layers from the downsampling path, 
      used for skip connections.
    - i: Integer index indicating which layer from conv_layers to concatenate with.
    - filter_list: List of integers, each specifying the number of filters for 
      the corresponding convolutional layer.
    - up_samp_size: Integer or tuple of 2 integers, specifying the upsampling 
      factor for rows and columns.
    - upsample_type: Boolean, whether to use transposed convolutions for 
      upsampling.

    Returns:
    - The output tensor after applying the upsampling block.
    """
    x = _upsampling_layer(x, filter_list[i], up_samp_size, upsample_type, 
                          batch_norm, kernel_initializer)
    
    x = Concatenate(axis=-1)([x, conv_layers[i]])

    x = conv_block(x, filter_size, filter_list[i], dropout_rate, num_conv_stack, 
                   batch_norm, kernel_initializer, SE_block, ratio, dilation_rate,
                   SE_pooling, SE_aggregation)
    
    return x


def upsample_runet(x, conv_layers, i, filter_list, filter_size, dropout_rate, 
                   num_conv_stack, batch_norm, kernel_initializer,  SE_block, 
                   up_samp_size, upsample_type, ratio, dilation_rate, 
                   SE_pooling, SE_aggregation):
    """
    Constructs an upsampling block for a modified U-Net architecture that includes 
    residual connections. This block upsamples the input feature maps and combines 
    them with feature maps from the downsampling path, followed by a residual 
    convolutional block.

    Parameters:
    Refer to the `upsample_unet` function for a detailed explanation of the parameters.
    """
    x = _upsampling_layer(x, filter_list[i], up_samp_size, upsample_type, 
                          batch_norm, kernel_initializer)
    
    x = Concatenate(axis=-1)([x, conv_layers[i]])

    x = res_conv_block(x, filter_size, filter_list[i], dropout_rate, num_conv_stack,
                       batch_norm, kernel_initializer, SE_block, ratio, dilation_rate,
                       SE_pooling, SE_aggregation)
    
    return x


def upsample_aunet(x, conv_layers, i, filter_list, filter_size, dropout_rate, 
                   num_conv_stack, batch_norm, kernel_initializer, SE_block, 
                   up_samp_size, upsample_type, ratio, dilation_rate, 
                   SE_pooling, SE_aggregation):
    """
    Constructs an upsampling block for a modified U-Net architecture that includes 
    a self-attention module. This block upsamples the input feature maps and combines 
    them with feature maps from the downsampling path through an attention mechanism.

    Parameters:
    Refer to the `upsample_unet` function for a detailed explanation of the parameters.
    """
    g = _upsampling_layer(x, filter_list[i], up_samp_size, upsample_type, 
                          batch_norm, kernel_initializer)  
     
    a = gated_attention(conv_layers[i], g, filter_list[i], kernel_initializer, batch_norm)

    x = Concatenate(axis=-1)([g, a])

    x = conv_block(x, filter_size, filter_list[i], dropout_rate, num_conv_stack, 
                   batch_norm, kernel_initializer, SE_block, ratio, dilation_rate,
                   SE_pooling, SE_aggregation)
    
    return x


def upsample_arunet(x, conv_layers, i, filter_list, filter_size, dropout_rate, 
                    num_conv_stack, batch_norm, kernel_initializer, SE_block, 
                    up_samp_size, upsample_type, ratio, dilation_rate, 
                    SE_pooling, SE_aggregation):
    """
    Constructs an upsampling block for a modified U-Net architecture that includes 
    both residual connections and a self-attention module. This block combines upsampled 
    input feature maps with feature maps from the downsampling path, enhanced by 
    attention and residual mechanisms.

    Parameters:
    Refer to the `upsample_unet` function for a detailed explanation of the parameters.
    """
    g = _upsampling_layer(x, filter_list[i], up_samp_size, upsample_type, 
                          batch_norm, kernel_initializer) 
      
    a = gated_attention(conv_layers[i], g, filter_list[i], kernel_initializer, batch_norm)
    
    x = Concatenate(axis=-1)([g, a])

    x = res_conv_block(x, filter_size, filter_list[i], dropout_rate, num_conv_stack,
                       batch_norm, kernel_initializer, SE_block, ratio, dilation_rate,
                       SE_pooling, SE_aggregation)
    
    return x


def final_conv_block(x, num_classes, kernel_initializer='he_normal', batch_norm=False, 
                     output_activation='sigmoid'):
    """
    Constructs the final convolutional block for various U-Net model variants. This block 
    outputs the final feature maps for the specified number of classes.

    Parameters:
    - x: Input tensor to the final convolutional block.
    - num_classes: Integer, the number of classes for the output layer. Determines 
      the number of filters.
    - kernel_initializer: Initializer for the kernel weights matrix.
    - batch_norm: Boolean, whether to include batch normalization in the block.
    - output_activation: String or keras.activations.Activation, the activation 
      function to use on the output layer

    The block consists of a 1x1 convolution that adjusts the number of output channels 
    to match the number of classes.
    """
    final_layer = Conv2D(num_classes, kernel_size=(1, 1), kernel_initializer=kernel_initializer)(x)
    if batch_norm:
        final_layer = BatchNormalization(axis=-1)(final_layer)
        
    final_layer = Activation(output_activation)(final_layer)
    
    return final_layer

