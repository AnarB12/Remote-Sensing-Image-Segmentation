import os
import numpy as np
import tifffile
from patchify import patchify
import shutil
from tqdm import tqdm


def validate_inputs(imgs_dir, masks_dir, patch_size):
    if not os.path.exists(imgs_dir) or not os.path.exists(masks_dir):
        raise ValueError("Input directories do not exist.")

    if patch_size <= 0:
        raise ValueError("Patch size must be a positive integer.")


def read_image(file_path):
    try:
        return tifffile.imread(file_path)
    except tifffile.TiffFileError:
        print(f"Error reading file: {file_path}")
        return None


def save_image(file_path, data):
    try:
        tifffile.imwrite(file_path, data)
    except tifffile.TiffFileError:
        print(f"Error saving file: {file_path}")


def patchify_images_and_masks(imgs_dir, masks_dir, output_imgs_dir, output_masks_dir, 
                              base_name="patch_", patch_size=256, num_channels=4, 
                              img_extension='.tif'):
    """
    Processes and patchifies images and masks from given directories.
    Saves the patchified images and masks into specified output directories.

    Args:
    - imgs_dir (str): Directory containing images.
    - masks_dir (str): Directory containing masks.
    - output_imgs_dir (str): Directory to save patchified images.
    - output_masks_dir (str): Directory to save patchified masks.
    - patch_size (int): Size of each patch (default 256).
    - num_channels (int): Number of channels in the images (default 4).
    """

    print("Starting the patchifying process...")

    global_counter = 0  

    img_counter = 0
    mask_counter = 0

    validate_inputs(imgs_dir, masks_dir, patch_size)

    def process_data(data_dir, save_dir, is_mask=False):
        nonlocal global_counter 
        nonlocal img_counter  
        nonlocal mask_counter  

        for _, _, files in os.walk(data_dir):
            data_files = [f for f in files if f.endswith(f'{img_extension}')]
            for data_name in tqdm(data_files, desc="Processing files"):
                data_path = os.path.join(data_dir, data_name)

                data = read_image(data_path)
                if data is None:
                    continue
                
                SIZE_X = (data.shape[1] // patch_size) * patch_size
                SIZE_Y = (data.shape[0] // patch_size) * patch_size

                data = data[:SIZE_Y, :SIZE_X]

                if len(data.shape) == 3:  # For multi-channel data
                    patch_shape = (patch_size, patch_size, num_channels)
                else:  # For single-channel data
                    patch_shape = (patch_size, patch_size)

                if is_mask:
                    mask_counter += 1
                else:
                    img_counter += 1

                # Create patches from the image
                patches = patchify(data, patch_shape, step=patch_size)

                # Iterate through the patches to save them
                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        single_patch = patches[i, j, :, :]
    
                        if is_mask:
                            single_patch = single_patch.astype('uint8')
                        else:
                            single_patch = single_patch[0]

                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
       
                        output_name = f"{base_name}{global_counter}{img_extension}"

                        save_image(os.path.join(save_dir, output_name), single_patch)
   
                        global_counter += 1


    process_data(imgs_dir, output_imgs_dir)
    global_counter = 0
    process_data(masks_dir, output_masks_dir, is_mask=True)

    print(f"Processed {img_counter} images and {mask_counter} masks.")
    print("Patchifying process completed.")


def copy_useful_images_and_masks(img_dir, mask_dir, output_img_dir, output_mask_dir, 
                                 threshold=0.10, image_dim=(512, 512), img_extension='.tif',
                                 remove_originals=False):
    """
    Copies images and their corresponding masks from given directories to output 
    directories if they meet certain criteria based on a threshold and image dimensions.

    This function processes images and masks in the specified directories, 
    checks whether the masks contain a sufficient amount of non-background 
    information (based on a threshold), and copies those images and masks 
    to the output directories if they do. Images are also checked for dimension 
    conformity and are skipped if they do not match the specified dimensions.

    Args:
    - img_dir (str): Directory containing source images.
    - mask_dir (str): Directory containing source masks.
    - output_img_dir (str): Directory to save useful images.
    - output_mask_dir (str): Directory to save useful masks.
    - threshold (float): Minimum required proportion of non-background pixels 
      in masks to consider them useful (default is 0.10).
    - image_dim (tuple): Expected dimensions of the images.
    - img_extension (str): File extension of the images and masks (default is '.tif').
    - remove_originals (bool): Whether to remove the original files after processing 
      (default is False).

    Raises:
    - ValueError: If there's a mismatch in the number of images and masks.
    """

    # print("Processing started...")
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    img_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                       if f.endswith(f'{img_extension}')])
    mask_list = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) 
                        if f.endswith(f'{img_extension}')])
    
    if len(img_list) != len(mask_list):
        raise ValueError("Mismatch in the number of images and masks.")

    useful_count = 0
    useless_count = 0
    skipped_count = 0
    
    for img_path, mask_path in tqdm(zip(img_list, mask_list), total=len(img_list), 
                                    desc="Processing images"):
        temp_image = tifffile.imread(img_path)
        temp_mask = tifffile.imread(mask_path)

        if temp_image.shape[:2] != image_dim:
            skipped_count += 1
            continue
        
        _, counts = np.unique(temp_mask, return_counts=True)
        
        if (1 - (counts[0] / counts.sum())) > threshold:
            tifffile.imwrite(os.path.join(output_img_dir, os.path.basename(img_path)), temp_image)
            tifffile.imwrite(os.path.join(output_mask_dir, os.path.basename(mask_path)), temp_mask)
            useful_count += 1
        else:
            useless_count += 1

    if remove_originals:
        for file in os.listdir(img_dir):
            os.remove(os.path.join(img_dir, file))
        for file in os.listdir(mask_dir):
            os.remove(os.path.join(mask_dir, file))

    print(f"Operation complete. {useful_count} useful image(s) and mask(s) were saved.")
    print(f"{useless_count} image(s) and mask(s) lack sufficient information and were skipped.")
    print(f"{skipped_count} image(s) were skipped due to mismatched dimensions.")


def combine_datasets(year1_path, year2_path, combined_path):
    """
    Combines two datasets from specified directories based on index, alternating 
    between the two datasets. For each index, if it is even, it takes the image and 
    mask from the first dataset; if odd, from the second dataset. This is intended
    to create a new dataset that interleaves items from the two source datasets.
    
    Directories within each dataset path should be structured into 'images' and 'masks'
    subdirectories. The function creates similar structured subdirectories in the
    specified combined path and copies the selected files accordingly.

    Args:
    - year1_path (str): Path to the directory of the first dataset. This directory
      must contain 'images' and 'masks' subdirectories.
    - year2_path (str): Path to the directory of the second dataset. This directory
      must also contain 'images' and 'masks' subdirectories.
    - combined_path (str): Path to the directory where the combined dataset should
      be saved. This directory will be created if it does not exist and will also
      contain 'images' and 'masks' subdirectories.

    Notes:
    - The function handles cases where the two datasets are not of the same size.
      It stops copying when it reaches the end of the smaller dataset.
    - Files are copied based on their sorted order in their respective directories.
    """

    os.makedirs(os.path.join(combined_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(combined_path, 'masks'), exist_ok=True)

    year1_images = sorted(os.listdir(os.path.join(year1_path, 'images')))
    year1_masks = sorted(os.listdir(os.path.join(year1_path, 'masks')))
    year2_images = sorted(os.listdir(os.path.join(year2_path, 'images')))
    year2_masks = sorted(os.listdir(os.path.join(year2_path, 'masks')))

    max_length = max(len(year1_images), len(year2_images))

    for i in tqdm(range(max_length), desc="Processing images and masks"):
        if i % 2 == 0:  # Even index
            if i < len(year1_images):
                shutil.copy(os.path.join(year1_path, 'images', year1_images[i]), 
                            os.path.join(combined_path, 'images', year1_images[i]))
                shutil.copy(os.path.join(year1_path, 'masks', year1_masks[i]), 
                            os.path.join(combined_path, 'masks', year1_masks[i]))
        else:  # Odd index
            if i < len(year2_images):
                shutil.copy(os.path.join(year2_path, 'images', year2_images[i]), 
                            os.path.join(combined_path, 'images', year2_images[i]))
                shutil.copy(os.path.join(year2_path, 'masks', year2_masks[i]), 
                            os.path.join(combined_path, 'masks', year2_masks[i]))



if __name__ == "__main__":    

    # Directories
    root_dir = r"F:\DEEP_LEARNING_GIS"
    imgs_dir = os.path.join(root_dir, "LandCover_Data", "images")
    masks_dir = os.path.join(root_dir, "LandCover_Data", "masks")
    output_imgs_dir = os.path.join(root_dir, "LandCover_Data_f", "images")
    output_masks_dir = os.path.join(root_dir, "LandCover_Data_f", "masks")

    # Patchify with a custom patch size, e.g., 512, 256, etc.,
    copy_useful_images_and_masks(imgs_dir, masks_dir, output_imgs_dir, output_masks_dir, 
                                 threshold=0.01, image_dim=(512, 512), img_extension='.TIF')

