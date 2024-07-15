import os
import numpy as np
import rasterio
from tifffile import imread
from keras.models import load_model
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm  
# from models import UNet, ResUNet, AttResUNet, scSEResUNet
from utils import scale_img
from custom_datagen import DataGenerator
scaler = MinMaxScaler()


def predict_big_image_with_blend(input_dir_path, model_file_path, patch_size, num_classes,
                                 subdivisions, calculate_pisi=False, calculate_ndvi=False,
                                 channels=None, threshold=0.5, output_file_path=None, 
                                 base_name="Prediction_"):

    """
    Predicts large images by dividing them into smaller patches and blending their 
    edges using `predict_img_with_smooth_windowing` function. 
    """

    model = load_model(model_file_path, compile=False)


    if output_file_path is not None and not os.path.exists(output_file_path):
        os.makedirs(output_file_path)


    tiff_files = [file for file in os.listdir(input_dir_path) if file.endswith('.TIF')]

    for i, file in tqdm(enumerate(tiff_files), total=len(tiff_files), desc="Processing Images"):
        full_file_path = os.path.join(input_dir_path, file)
        image = imread(full_file_path)

        # Apply PISI and NDVI index calculations if enabled
        if calculate_pisi:
            PISI = DataGenerator.calculate_pisi_(image, channels)
            image = np.concatenate((image, PISI), axis=-1) 

        if calculate_ndvi:
            NDVI = DataGenerator.calculate_ndvi_(image, channels)
            image = np.concatenate((image, NDVI), axis=-1)

        # Scale image
        image = scale_img(image, scaler=scaler)

        # Make predictions using smooth windowing
        predictions_smooth = predict_img_with_smooth_windowing(
            image,
            window_size=patch_size,
            subdivisions=subdivisions,
            nb_classes=num_classes,
            pred_func=(
                lambda img_batch_subdiv: model.predict(img_batch_subdiv, verbose=0)
            )
        )

        if num_classes == 1:
            prediction = (predictions_smooth > threshold).astype('uint8')
        else:
            prediction = np.argmax(predictions_smooth, axis=2).astype('uint8')


        if output_file_path is not None:
            output_file_name = f"{base_name}{i}.tif"
            output_full_path = os.path.join(output_file_path, output_file_name)
            save_prediction_with_metadata(full_file_path, prediction, output_full_path)

    print("Process finished.")



def save_prediction_with_metadata(input_image_path, prediction, output_file_path):
    """
    Saves the prediction with spatial metadata extracted from the input image.
    """
    if output_file_path is None:
        raise ValueError("No output path provided for saving the prediction.")

    with rasterio.open(input_image_path) as src:
        meta = src.meta

    meta.update(dtype=rasterio.uint8, count=1, compress='lzw')

    with rasterio.open(output_file_path, 'w', **meta) as dst:
        dst.write(prediction.squeeze(), 1)


#=========================================================================================
#=========================================================================================


def predict_patches(input_dir_path, model_file_path, num_classes, img_size=(512, 512),
                    calculate_pisi=False, calculate_ndvi=False,channels=None, 
                    threshold=0.5, output_file_path=None, augment=False, 
                    base_name="Prediction_", img_extension=".TIF"):


    def rotate_and_mirror_image(img):
        rotations = [np.rot90(img, k=i) for i in range(4)]
        mirrored = [np.fliplr(x) for x in rotations]
        return rotations + mirrored


    def undo_transformations(predictions):
        untransformed_predictions = []
        for i, pred in enumerate(predictions):
            if i < 4: 
                untransformed_predictions.append(np.rot90(pred, k=-i))
            else:
                untransformed_predictions.append(np.rot90(np.fliplr(pred), k=-(i - 4)))

        return untransformed_predictions


    def pad_image(image, target_size):
        height, width = image.shape[:2]
        delta_w = target_size[1] - width
        delta_h = target_size[0] - height
        padding = ((0, delta_h), (0, delta_w), (0, 0)) if len(image.shape) == 3 else ((0, delta_h), (0, delta_w))
        padded_image = np.pad(image, padding, mode='constant', constant_values=0)
        return padded_image, padding


    def unpad_image(image, padding):
        if len(image.shape) == 3:
            return image[:image.shape[0] - padding[0][1], :image.shape[1] - padding[1][1], :]
        else:
            return image[:image.shape[0] - padding[0][1], :image.shape[1] - padding[1][1]]


    model = load_model(model_file_path, compile=False)


    tiff_files = [file for file in os.listdir(input_dir_path) if file.endswith(f"{img_extension}")]


    for i, file in tqdm(enumerate(tiff_files), total=len(tiff_files), desc="Processing Images"):
        full_file_path = os.path.join(input_dir_path, file)
        image = imread(full_file_path)
        # original_size = image.shape[:2]

        # Apply PISI and NDVI index calculations if enabled
        if calculate_pisi:
            PISI = DataGenerator.calculate_pisi_(image, channels)
            image = np.concatenate((image, PISI), axis=-1) 

        if calculate_ndvi:
            NDVI = DataGenerator.calculate_ndvi_(image, channels)
            image = np.concatenate((image, NDVI), axis=-1)

        # Scale image
        image = scale_img(image, scaler=scaler)

        # Check image size and pad if necessary
        if image.shape[:2] != img_size:
            image, padding = pad_image(image, img_size)
        else:
            padding = ((0, 0), (0, 0))

        if augment and num_classes == 1:
            # Apply transformations
            transformed_images = rotate_and_mirror_image(image)
            
   
            transformed_predictions = []
            for transformed_img in transformed_images:
                transformed_img_batch = np.expand_dims(transformed_img, axis=0) 
                transformed_pred_mask = model(transformed_img_batch, training=False)
                transformed_predictions.append(np.squeeze(transformed_pred_mask))
            
            # Undo the transformations
            aligned_predictions = undo_transformations(transformed_predictions)
            
            # Average the predictions
            avg_pred_mask = np.mean(aligned_predictions, axis=0)
        else:
            # Predict the original image
            img_batch = np.expand_dims(image, axis=0)
            avg_pred_mask = model(img_batch, training=False)
            avg_pred_mask = np.squeeze(avg_pred_mask)

        if num_classes == 1:
            prediction = (avg_pred_mask > threshold).astype('uint8')
        else:
            prediction = np.argmax(avg_pred_mask, axis=2).astype('uint8')

        # Unpad the prediction to the original image size
        prediction = unpad_image(prediction, padding)

        if output_file_path is not None:
            if base_name:
                output_file_name = f"{base_name}{i}{img_extension}"
            else:
                output_file_name = f"{os.path.splitext(file)[0]}_pred{img_extension}"
            output_full_path = os.path.join(output_file_path, output_file_name)
            save_prediction_with_metadata(full_file_path, prediction, output_full_path)

    print("Predictions finished.")



if __name__ == "__main__":

    input_dir_path = r"F:\DEEP_LEARNING_GIS\Predictions\2023_6in_AOI_LULC_512\Image_patches"
    model_file_path = r"F:\DEEP_LEARNING_GIS\Trained_Models\UNet_LandCover_v1.hdf5"
    output_file_path = r"F:\DEEP_LEARNING_GIS\Predictions\2023_6in_AOI_LULC_512\Pred_patches"
    channels = {'R': 0, 'G': 1, 'B': 2, 'NIR': 3}
    # Define parameters
    # patch_size = 1024  # patch size
    num_classes = 5   # number of classes
    img_size=(512, 512)
    # subdivisions = 2  # pixel overlap
    
    # # Call the function with the parameters
    # predict_big_image_with_blend(input_dir_path, model_file_path, patch_size, num_classes,
    #                              subdivisions, calculate_pisi=False, calculate_ndvi=False,
    #                              channels=None, threshold=0.5, output_file_path=output_file_path, 
    #                              base_name="Prediction_")

    img_extension = ".TIF"

    predict_patches(input_dir_path, 
                    model_file_path, 
                    num_classes, 
                    img_size=img_size,
                    calculate_pisi=True, # if the model was trained on images with these indices calculated,
                    calculate_ndvi=True, # enable them accordingly as the model expects the same input channel
                    channels=channels, 
                    threshold=None,
                    output_file_path=output_file_path, 
                    augment=False, 
                    base_name=None,
                    img_extension=img_extension)

