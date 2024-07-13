import os
import numpy as np
from tifffile import imread
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import albumentations as A


class DataGenerator:
    """
    A Data generator for loading and preprocessing batches of images 
    and masks for training in U-Net model variants.

    This function supports various functionalities including image augmentation, 
    normalization, categorical conversion of masks, calculation of specific 
    indices like PISI and NDVI, and compatibility with k-fold cross-validation.

    Attributes:
    - img_dir (str): Directory containing the training images.
    - mask_dir (str): Directory containing the corresponding masks.
    - batch_size (int): Size of the batches of data.
    - n_classes (int): Number of classes for segmentation.
    - image_shape (tuple): Shape of the input images.
    - image_aug (bool): Flag to enable/disable image augmentation.
    - index_shuffle (bool): Flag to enable/disable shuffling of the data.
    - channel_shuffle (bool): Flag to enable/disable shuffling the channels.
    - calculate_pisi (bool): Flag to enable/disable calculating PISI.
    - calculate_ndvi (bool): Flag to enable/disable NDVI calculation.
    - k_fold_indices (list/array, optional): List of indices for each fold in 
      k-fold cross-validation. Each sublist in this list represents the indices 
      of images and masks to be used in one fold of the validation. These indices 
      should correspond to the order of images and masks as they appear in their 
      respective directories. When set, the DataGenerator will only load images 
      and masks corresponding to the indices in the currently active fold.
    - img_extension (str): Extension of the images in the dataset.

    Methods:
    - preprocess_img(image): Normalizes and preprocesses an input image.
    - calculate_pisi(image): Calculates the Perpendicular Impervious Surface Index 
      (PISI) using the NIR and Blue channels.
    - calculate_ndvi(image): Calculates the Normalized Difference Vegetation Index 
      (NDVI) using NIR and Red channels.
    - load_img_and_mask(img_indices): Loads, preprocesses, augments images and masks, 
      and calculates indices like PISI and NDVI based on provided indices.
    - imageLoader(): Generator function to yield batches of images and masks, and
      handling data shuffling if enabled.

    Raises:
    - FileNotFoundError: If the specified image or mask directory does not exist.
      ValueError: If no images/masks are found in the provided directories or 
      if the number of images and masks is not equal.
    """
    def __init__(self, img_dir, mask_dir, batch_size=8, n_classes=1, image_shape=(512, 512), 
                 image_aug=False, index_shuffle=False, channel_shuffle=False, calculate_pisi=False, 
                 calculate_ndvi=False, k_fold_indices=None, img_extension='.tif', channels=None):

        if channels is None:
            channels = {'R': 0, 'G': 1, 'B': 2, 'NIR': 3}
            
        self.validate_directories(img_dir, mask_dir)
        self.channels = channels
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.image_shape = image_shape
        self.image_aug = image_aug
        self.shuffle = index_shuffle
        self.channel_shuffle = channel_shuffle
        self.calculate_pisi = calculate_pisi
        self.calculate_ndvi = calculate_ndvi
        self.scaler = MinMaxScaler()
        self.img_list = sorted([f for f in os.listdir(img_dir) if f.endswith(f'{img_extension}')])
        self.mask_list = sorted([f for f in os.listdir(mask_dir) if f.endswith(f'{img_extension}')])

        if k_fold_indices is not None:
            # Filter the img_list and mask_list based on the provided indices
            # Using sklearn kfold object that return indices of the splits
            self.img_list = [self.img_list[i] for i in k_fold_indices]
            self.mask_list = [self.mask_list[i] for i in k_fold_indices]

        self.validate_lists()

        if self.image_aug:
            self.augmentation_pipeline = self.def_augmentation_pipeline()

        self.generator = self.imageLoader()


    def def_augmentation_pipeline(self):
        """
        If augmentations are enabled, define the augmentation pipeline.
        Implemented `Albumentation Library` for image transformation. 
        Add more augmentations if needed. Please refer to the following
        link: https://albumentations.ai/docs/
        """
        return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=(-90, 90), p=0.2),
                A.RandomSizedCrop(min_max_height=(
                                  int(self.image_shape[0] * 0.85), 
                                      self.image_shape[0]
                                  ),
                                  height=self.image_shape[0], 
                                  width=self.image_shape[0], 
                                  p=0.3)            
            ], 
            p=1)
    

    def validate_directories(self, img_dir, mask_dir):
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            raise FileNotFoundError("Image or Mask directory does not exist.")


    def validate_lists(self):
        if not self.img_list or not self.mask_list:
            raise ValueError(
                "No image or mask files found in the provided directories."
            )
        if len(self.img_list) != len(self.mask_list):
            raise ValueError("Number of images and masks should be the same.")


    def scale_img(self, image):
        """
        Preprocesses the images. Scales image channel individually between 0 and 1 .
        Additionally, clips values to ensure they stay within the [0.0, 1.0] range.
        """
        reshaped_image = image.reshape(-1, image.shape[-1])
        scaled_image = self.scaler.fit_transform(reshaped_image).reshape(image.shape)
        # Clip values to ensure they are within [0.0, 1.0]
        clipped_image = np.clip(scaled_image, 0.0, 1.0)

        return clipped_image


    @staticmethod
    def calculate_pisi_(image, channels):
        """
        Calculate the Perpendicular Impervious Surface Index (PISI) 
        using the NIR channel. PISI = 0.8191 * Blue - 0.5735 * NIR + 0.075
        https://www.mdpi.com/2072-4292/10/10/1521 
        """
        B_channel = image[:, :, channels['B']]  # Blue channel
        NIR_channel = image[:, :, channels['NIR']]  # NIR channel
        PISI = 0.8191 * B_channel - 0.5735 * NIR_channel + 0.075
        PISI = np.expand_dims(PISI, axis=-1)
        return PISI.astype('float32')


    @staticmethod
    def calculate_ndvi_(image, channels):
        """
        Calculate the Normalized Difference Vegetation Index (NDVI) 
        using NIR and Red channels. NDVI = (NIR - Red) / (NIR + Red)
        """
        R_channel = image[:, :, channels['R']]  # Red channel
        NIR_channel = image[:, :, channels['NIR']]  # NIR channel
        with np.errstate(divide='ignore', invalid='ignore'):
            NDVI = (NIR_channel - R_channel) / (NIR_channel + R_channel)
            NDVI = np.nan_to_num(NDVI, nan=-1)
        NDVI = np.expand_dims(NDVI, axis=-1)
        return NDVI.astype('float32')


    def load_img_and_mask(self, img_indices):
        """
        Loads a batch of images and masks based on the provided indices.
        Applies preprocessing, augmentation (if enabled), and calculates 
        indices like PISI and NDVI.
        """
        images = []
        masks = []
        
        for idx in img_indices:
            img_path = os.path.join(self.img_dir, self.img_list[idx])
            mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

            image = imread(img_path)
            mask = imread(mask_path)

            if self.calculate_pisi:
                PISI = DataGenerator.calculate_pisi_(image, self.channels)
                image = np.concatenate((image, PISI), axis=-1)

            if self.calculate_ndvi:
                NDVI = DataGenerator.calculate_ndvi_(image, self.channels)
                image = np.concatenate((image, NDVI), axis=-1)

            if self.channel_shuffle:
                channel_shuffle_transform = A.ChannelShuffle()
                image = channel_shuffle_transform(image=image)["image"]

            if self.image_aug:
                augmented = self.augmentation_pipeline(image=image, mask=mask)
                image, mask = augmented["image"], augmented["mask"]

            processed_image = self.scale_img(image)

            if self.n_classes == 1:
                processed_mask = np.expand_dims(mask, axis=-1)
            else:
                processed_mask = to_categorical(mask, self.n_classes)

            images.append(processed_image)
            masks.append(processed_mask)

        img_arrays = np.array(images).astype('float32')
        mask_arrays = np.array(masks).astype('float32')

        return img_arrays, mask_arrays


    def imageLoader(self):
        """
        Generator function that yields batches of images and masks. Handles shuffling 
        of data if enabled. Returns a tuple of image and mask arrays for each batch.
        """        
        total_images = len(self.img_list)

        while True:
            if self.shuffle:
                shuffled_indices = np.random.permutation(total_images)
            else:
                shuffled_indices = np.arange(total_images)
            
            for start_idx in range(0, total_images, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_images)
                current_batch_indices = shuffled_indices[start_idx:end_idx]

                # Process the images and masks in the batch
                batch_images, batch_masks = self.load_img_and_mask(current_batch_indices)
                
                # Yield the loaded batch of images and masks 
                yield (batch_images, batch_masks)
                                                  
      