import numpy as np

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


if __name__ == "__main__":
    
    original_image = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
    ])

    # Target size for padding
    target_size = (5, 5)

    # Pad the image
    padded_image, padding = pad_image(original_image, target_size)

    # Unpad the image
    unpadded_image = unpad_image(padded_image, padding)

    print("Original Image:\n", original_image)
    print("Padded Image:\n", padded_image)
    print("Unpadded Image:\n", unpadded_image)
