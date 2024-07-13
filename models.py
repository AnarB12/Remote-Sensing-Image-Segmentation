import tensorflow as tf
from keras.models import Model
from keras.layers import Input 
from blocks import (conv_block, res_conv_block, 
                    upsample_unet, upsample_runet, 
                    upsample_aunet, upsample_arunet)
from blocks import (final_conv_block, 
                    bottleneck_block, 
                    downsampling_block)


def build_unet_base(input_shape, filter_list, num_classes, dropout_rate, 
                    num_conv_stack, batch_norm, kernel_initializer, 
                    SE_block_type, SE_ratio, SE_pooling, SE_aggregation, 
                    upsample_type, output_activation, dilation_rate, 
                    bottleneck_type, atrous_rates, downsample_type, 
                    conv_block_type, upsample_block_type, model_name):
    """
    Constructs the base architecture for various U-Net variants. This function 
    serves as a flexible and customizable foundation for creating different 
    versions of the U-Net architecture. 
    

    Parameters like filter sizes, dropout rates, convolution layers, and specific 
    components like squeeze-and-excitation (SE) blocks, atrous spatial pyramid 
    pooling (ASPP), and various types of bottleneck layers are adjustable. 
    This enables the construction of both standard and advanced U-Net models 
    with specific requirements.


    It also allows for varying the downsampling and upsampling strategies, 
    which are important in capturing context and enabling  localization in 
    U-Net architectures.
    """

    inputs = Input(input_shape, dtype=tf.float32)
    x = inputs
    FILTER_SIZE = 3
    UPSAMPLE_SIZE = 2
    POOL_SIZE = 2
    conv_layers = []

    # Downsampling layers (Contraction path)
    # Note: Exclude the last element of filter_list for downsampling,
    # because it's now intended for the bottleneck layer.
    for filters in filter_list[:-1]:  
        # Apply scSE if enabled
        use_se_block = SE_block_type in ['both', 'down']

        x = conv_block_type(x, FILTER_SIZE, filters, dropout_rate, num_conv_stack, 
                            batch_norm, kernel_initializer, use_se_block, SE_ratio, 
                            dilation_rate, SE_pooling, SE_aggregation)
        conv_layers.append(x)

        x = downsampling_block(x, downsample_type, POOL_SIZE, filters, 
                               FILTER_SIZE, batch_norm, kernel_initializer)

    # Bottleneck layer
    # Directly use the last filter size in filter_list for the bottleneck
    bottleneck_filters = filter_list[-1]  
    x = bottleneck_block(x, conv_block_type, bottleneck_type, bottleneck_filters, 
                         FILTER_SIZE, dropout_rate, num_conv_stack, batch_norm, 
                         kernel_initializer, atrous_rates)

    # Upsampling layers (Expansive path)
    # Note: Start with the second last filter size as the first one is used 
    #       for the bottleneck
    for i in reversed(range(len(filter_list) - 1)): 
        # Apply scSE if enabled
        use_se_block = SE_block_type in ['both', 'up']

        x = upsample_block_type(x, conv_layers, i, filter_list, FILTER_SIZE, 
                                dropout_rate, num_conv_stack, batch_norm, 
                                kernel_initializer, use_se_block, UPSAMPLE_SIZE, 
                                upsample_type, SE_ratio, 1, SE_pooling, SE_aggregation)

    # Final convolution layer
    x = final_conv_block(x, num_classes, kernel_initializer, batch_norm, output_activation)

    # Model instance
    model = Model(inputs=[inputs], outputs=[x], name=model_name)
    return model


def UNet(model_name, input_shape, filter_list=[64, 128, 256, 512], 
         num_classes=1, dropout_rate=0.0, num_conv_stack=2, 
         batch_norm=True, kernel_initializer='he_normal', 
         upsample_type='bilinear', output_activation=None, 
         dilation_rate=1, bottleneck_type='conv', 
         atrous_rates=None, downsample_type='maxpool'):
    """
    Constructs a standard U-Net model with specified parameters.
    """    
    return build_unet_base(input_shape, filter_list, num_classes, 
                           dropout_rate, num_conv_stack, batch_norm,
                           kernel_initializer, None, None, None, None, 
                           upsample_type, output_activation, 
                           dilation_rate, bottleneck_type, atrous_rates, 
                           downsample_type, conv_block, upsample_unet, 
                           model_name=model_name)


def ResUNet(model_name, input_shape, filter_list=[64, 128, 256, 512], 
            num_classes=1, dropout_rate=0.0, num_conv_stack=2, 
            batch_norm=True, kernel_initializer='he_normal', 
            upsample_type='bilinear', output_activation=None, 
            dilation_rate=1, bottleneck_type='conv', 
            atrous_rates=None, downsample_type='maxpool'):
    """
    Constructs a Residual U-Net (ResUNet) model with specified 
    parameters.
    """    
    return build_unet_base(input_shape, filter_list, num_classes, 
                           dropout_rate, num_conv_stack, batch_norm, 
                           kernel_initializer, None, None, None, 
                           None, upsample_type, output_activation, 
                           dilation_rate, bottleneck_type, atrous_rates, 
                           downsample_type, res_conv_block, upsample_runet, 
                           model_name=model_name)


def AttUNet(model_name, input_shape, filter_list=[64, 128, 256, 512], 
            num_classes=1, dropout_rate=0.0, num_conv_stack=2, 
            batch_norm=True, kernel_initializer='he_normal',
            upsample_type='bilinear', output_activation=None, 
            dilation_rate=1, bottleneck_type='conv', 
            atrous_rates=None, downsample_type='maxpool'):
    """
    Constructs an Attention U-Net (AttUNet) model with specified 
    parameters.
    """    
    return build_unet_base(input_shape, filter_list, num_classes, 
                           dropout_rate, num_conv_stack, batch_norm,
                           kernel_initializer, None, None, None,None, 
                           upsample_type, output_activation, 
                           dilation_rate, bottleneck_type, atrous_rates, 
                           downsample_type, conv_block, upsample_aunet, 
                           model_name=model_name)


def AttResUNet(model_name, input_shape, filter_list=[64, 128, 256, 512],
               num_classes=1, dropout_rate=0.0, num_conv_stack=2, 
               batch_norm=True, kernel_initializer='he_normal', 
               upsample_type='bilinear',output_activation=None, 
               dilation_rate=1, bottleneck_type='conv', 
               atrous_rates=None, downsample_type='maxpool'):
    """
    Constructs an Attention Residual U-Net (AttResUNet) model with 
    specified parameters.
    """
    return build_unet_base(input_shape, filter_list, num_classes, 
                           dropout_rate, num_conv_stack, batch_norm,
                           kernel_initializer, None, None, None, None, 
                           upsample_type, output_activation, 
                           dilation_rate, bottleneck_type, atrous_rates, 
                           downsample_type, res_conv_block, upsample_arunet, 
                           model_name=model_name)


def scSEUNet(model_name, input_shape, filter_list=[64, 128, 256, 512], 
             num_classes=1, dropout_rate=0.0, num_conv_stack=2, 
             batch_norm=True, kernel_initializer='he_normal', 
             SE_block_path='both', SE_ratio=8, SE_pooling='avg', 
             SE_aggregation='concatenate', upsample_type='bilinear', 
             output_activation=None, dilation_rate=1, bottleneck_type='conv', 
             atrous_rates=None, downsample_type='maxpool'):
    """
    Constructs a scSE U-Net (scSEUNet) model with specified parameters. 
    scSE refers to spatial and channel 'Squeeze and Excitation'.
    """
    return build_unet_base(input_shape, filter_list, num_classes, 
                           dropout_rate, num_conv_stack, batch_norm, 
                           kernel_initializer, SE_block_path, 
                           SE_ratio, SE_pooling, SE_aggregation, 
                           upsample_type, output_activation,
                           dilation_rate, bottleneck_type, atrous_rates, 
                           downsample_type, conv_block, upsample_unet, 
                           model_name=model_name)


def scSEResUNet(model_name, input_shape, filter_list=[64, 128, 256, 512], 
                num_classes=1, dropout_rate=0.0, num_conv_stack=2, 
                batch_norm=True,kernel_initializer='he_normal', 
                SE_block_path='both', SE_ratio=8, SE_pooling='avg', 
                SE_aggregation='concatenate', upsample_type='bilinear', 
                output_activation=None, dilation_rate=1, bottleneck_type='conv', 
                atrous_rates=None, downsample_type='maxpool'):
    """
    Constructs a scSE Residual U-Net (scSEResUNet) model with specified 
    parameters. This model combines spatial and channel 'Squeeze and 
    Excitation' (scSE) blocks with residual connections in the U-Net 
    architecture.
    """    
    return build_unet_base(input_shape, filter_list, num_classes, 
                           dropout_rate, num_conv_stack, batch_norm, 
                           kernel_initializer, SE_block_path, SE_ratio, 
                           SE_pooling, SE_aggregation, upsample_type, 
                           output_activation, dilation_rate, bottleneck_type, 
                           atrous_rates, downsample_type, res_conv_block, 
                           upsample_runet, model_name=model_name)


def get_model(model_name, input_shape, **kwargs):
    """
    Creates and returns a U-Net type segmentation model based on the specified 
    model_name and input_shape, with additional configurable parameters.

    Supported model names:
    - "unet": Standard U-Net architecture.
    - "resunet": U-Net architecture with residual connections.
    - "attention_unet": U-Net architecture with attention mechanisms.
    - "attention_resunet": U-Net architecture with both attention mechanisms
       and residual connections.
    - "scse_unet": U-Net architecture with Squeeze-and-Excitation (SE) blocks.
    - "scse_resunet": U-Net architecture with Squeeze-and-Excitation (SE) blocks 
       and residual connections.

    Parameters:
    - model_name (str): Name of the model to be created.
    - input_shape (tuple): Shape of the input data.

    Optional Keyword Arguments (kwargs):
    - filter_list (list): List of filters for each convolutional layer.
    - num_classes (int): Number of classes for the output layer.
    - num_conv_stack (int): Number of convolutional layers in the block.
    - batch_norm (bool): Whether to use batch normalization.
    - upsample_type (str): Whether to use transposed convolutions.
    - output_activation (str): Activation function for the output layer.
    - bottleneck_type (str): Type of bottleneck layer. Options are 'aspp', 
      'conv', 'conv+aspp'.
    - atrous_rates (list): Atrous rates for dilated convolutions in the bottleneck.
    - downsample_type (str): Type of downsampling layer. Options are 'conv','maxpool'
    - dropout_rate (float): Rate of dropout to apply to the model.
    - kernel_initializer (str): Initializer for the kernel weights.
    - dilation_rate (int): Dilation rate for convolution layers. If a single 
      integer, the same value is used for all layers. If a list, each element 
      is applied to the corresponding convolution layer in the block.
    - SE_block_path (str): Path for SE blocks, applicable for 'scse_unet' 
      and 'scse_resunet'. Options are 'down', 'up' or 'both'
    - SE_ratio (int): Ratio for SE blocks, applicable for 'scse_unet' and 
      'scse_resunet'. 
    - SE_pooling (str): Pooling type for SE blocks, applicable for 'scse_unet' 
      and 'scse_resunet'. Type of pooling for spatial squeeze. Options are 'avg', 
      'max', or 'both'. If 'both', average and max pooled values are added.
    - SE_aggregation (str): Aggregation method for SE blocks, applicable for
      'scse_unet' and 'scse_resunet'. Method for aggregating spatial and 
      channel squeeze outputs. Options are 'max', 'add', 'multiply', 'concatenate'.

    Returns:
    An instance of the specified model with the provided configurations.

    Raises:
    - ValueError: If the provided model_name is not supported.
    """
    common_params = {
        "input_shape": input_shape,
        "filter_list": [64, 128, 256, 512],
        "num_classes": 1,
        "num_conv_stack": 2,
        "batch_norm": True,
        "upsample_type": 'deconvolution',
        "output_activation": 'sigmoid',
        "bottleneck_type": 'conv',
        "atrous_rates": None,
        "downsample_type": 'maxpool',
        "dropout_rate": 0.0, 
        "kernel_initializer": 'he_normal', 
        "dilation_rate": 1,
    }

    common_params.update(kwargs)

    if model_name == "unet":
        return UNet(model_name=model_name, **common_params)
    elif model_name == "resunet":
        return ResUNet(model_name=model_name, **common_params)
    elif model_name == "attention_unet":
        return AttUNet(model_name=model_name, **common_params)
    elif model_name == "attention_resunet":
        return AttResUNet(model_name=model_name, **common_params)
    elif model_name == "scse_unet":
        scse_params = {
            "model_name": model_name, 
            "SE_block_path": 'both', 
            "SE_ratio": 4, 
            "SE_pooling": 'avg', 
            "SE_aggregation": 'add'
        }
        scse_params.update(common_params)
        return scSEUNet(**scse_params)
    elif model_name == "scse_resunet":
        scse_resunet_params = {
            "model_name": model_name,
            "SE_block_path": 'both', 
            "SE_ratio": 4, 
            "SE_pooling": 'avg', 
            "SE_aggregation": 'add'
        }
        scse_resunet_params.update(common_params)
        return scSEResUNet(**scse_resunet_params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

