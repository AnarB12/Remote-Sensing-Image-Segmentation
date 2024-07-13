import os
import numpy as np
import tifffile
from scipy import stats
from tqdm import tqdm


def is_even(pixel):
    return pixel % 2 == 0 and pixel != 0


def get_odd_neighbors(padded_image, x, y, window_size):
    half_window = window_size // 2
    odd_values = []

    # Adjust the coordinates for the padded image
    x += half_window
    y += half_window

    for i in range(x - half_window, x + half_window + 1):
        for j in range(y - half_window, y + half_window + 1):
            value = padded_image[i, j]
            if not is_even(value):
                odd_values.append(value)
                    
    return odd_values


def calculate_mode(values):
    if values:
        mode_val, _ = stats.mode(values, axis=None)
        return int(mode_val[0])
    return None


def correct_mask(image, window_size=5):
    half_window = window_size // 2
    # Pad the image to handle edge cases
    padded_image = np.pad(image, pad_width=half_window, 
                          mode='constant', constant_values=0)
    corrected = np.copy(image)
    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            if is_even(image[i, j]):
                # Adjust coordinates to account for the padding in `padded_image`
                odd_neighbors = get_odd_neighbors(padded_image, 
                                                  i, j, window_size)
                mode_value = calculate_mode(odd_neighbors)

                if mode_value is not None: 
                    corrected[i, j] = mode_value

    return corrected


def process_masks(mask_path, window_size=5, mask_extension='.tif'):

    corrected_dir = os.path.join(mask_path, 'corrected_masks')
    os.makedirs(corrected_dir, exist_ok=True)

    for filename in tqdm(os.listdir(mask_path), desc="Processing Masks"):
        if filename.endswith(mask_extension):
            file_path = os.path.join(mask_path, filename)
            image = tifffile.imread(file_path)

            corrected = correct_mask(image, window_size)
            corrected_file_path = os.path.join(corrected_dir, filename)

            tifffile.imwrite(corrected_file_path, corrected)

    print("Done processing all masks.")
