# image_utils.py
# image_utils.py
from .common_imports import *
from .common_utils import *
from .data_utils import *
from .evaluation_utils import *
from .model_utils import *
from .training_utils import *
from .visualization_utils import *


import numpy as np
import cv2

def bezier_curve(points, n=1000):
    N = len(points)
    x_vals, y_vals = np.array([0.0] * n), np.array([0.0] * n)
    t = np.linspace(0, 1, n)

    for i in range(N):
        x_vals += binom(N - 1, i) * ((1 - t) ** (N - 1 - i)) * (t ** i) * points[i][0]
        y_vals += binom(N - 1, i) * ((1 - t) ** (N - 1 - i)) * (t ** i) * points[i][1]

    return list(zip(x_vals.astype(int), y_vals.astype(int)))

def binom(n, k):
    return np.math.factorial(n) // (np.math.factorial(k) * np.math.factorial(n - k))

def createMaskv1(img, add_noise=True, thickness=2):
    if np.ndim(img) == 3:
        img = np.squeeze(img)

    h, w = np.shape(img)
    mask = np.full((h, w), 0, np.uint8)
    if np.sum(img) < 300 or np.sum(img) > 202144.0:
       #print('zero')
       return np.array(img)
    else:
      #print('edge')

      for _ in range(int(np.clip(np.random.normal(20, 7), 0, 40))):  # Mean is 10, standard deviation is 7
          control_points = [(np.random.randint(0, h), np.random.randint(0, w)) for _ in range(4)]
          points = bezier_curve(control_points)

          for point in points:
              x, y = point
              if 0 <= x < h and 0 <= y < w:
                  cv2.circle(mask, (y, x), thickness, (1), -1)

      # if add_noise:
      #     noise = np.random.normal(0, 0.2, mask.shape)
      #     mask = np.clip(mask + noise, 0, 1)
      idx = (mask > 0)
      img[idx] = 0
      if add_noise:
        if np.random.rand() < 0.65:

          # Adjust Gaussian noise std based on image range
          range_value = np.max(img) - np.min(img)
          #print(range_value)
          scalar = random.uniform(0.01, 0.2)
          adjusted_std = scalar * range_value
          img = add_gaussian_noise(img, std=adjusted_std)

      return np.array(img)



def add_artifacts(image, num_dots=30, dot_size=3, blur_size=3, **kwargs):
    """
    Add small white artifacts on the image.

    Parameters:
    - image: The input image to modify.
    - num_dots: Number of dots/artifacts to add.
    - dot_size: Size of the dots.
    - blur_size: Size of Gaussian blur kernel to blur the artifacts.

    Returns:
    - Modified image with artifacts.
    """

    modified_image = image.copy()  # Work on a copy to preserve the original image
    height, width = modified_image.shape[:2]
    max_val = np.max(modified_image)  # Get the maximum pixel value in the image
    max_val = 1
    for _ in range(num_dots):
        # Randomly select a center for the dot
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)

        # Draw the dot with the intensity set to the maximum pixel value of the image
        cv2.circle(modified_image, (center_x, center_y), dot_size, (max_val, max_val, max_val), -1)

    # Blur the image slightly to make the artifacts softer
    blurred_image = cv2.GaussianBlur(modified_image, (blur_size, blur_size), 0)

    # Ensure the image has 3 dimensions
    if len(blurred_image.shape) == 2:
        blurred_image = np.expand_dims(blurred_image, -1)

    return blurred_image
import numpy as np
import albumentations as A
import random

def aug_batch_segconnect(batch_y):
    """
    The 'aug_batch_segconnect' function applies simplified image augmentations
    specifically flipping and Gaussian noise to input mask list, 'batch_y'.
    It returns the augmented and original mask lists.
    """

    yn_original = []
    yn_augmented = []

    for mask in batch_y:

        alpha = random.uniform(30, 60)
        sigma = alpha * random.uniform(0.02, 0.07)
        alpha_affine = alpha * random.uniform(0.02, 0.05)

        original_height, original_width = mask.shape[:2]

        # Decide random crop size
        crop_height = int(original_height * random.uniform(0.6, 1))
        crop_width = int(original_width * random.uniform(0.6, 1))

        # Calculate random start coordinates for the cropping to ensure we get the same crop
        start_x = np.random.randint(0, original_width - crop_width + 1)
        start_y = np.random.randint(0, original_height - crop_height + 1)

        end_x = start_x + crop_width
        end_y = start_y + crop_height

        # Apply cropping based on calculated coordinates
        mask_cropped = mask[start_y:end_y, start_x:end_x]
        mask_cropped_resized = cv2.resize(mask_cropped, (original_width, original_height))

        m_aug = mask.copy()
        dots = int(random.uniform(10, 50))
        if np.random.rand() < 0.9:
            m_aug = add_artifacts(m_aug, num_dots=dots, dot_size=3, blur_size=3)

        m_aug_cropped = m_aug[start_y:end_y, start_x:end_x]
        m_aug_cropped_resized = cv2.resize(m_aug_cropped, (original_width, original_height))

        # Apply the random ElasticTransform
        elastic_transform = A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, p=1)

        if np.random.rand() < 0.4:
          elastic_mask = mask_cropped_resized
          elastic_aug = elastic_transform(image=m_aug_cropped_resized)["image"]
        else:
          elastic_mask = mask_cropped_resized
          elastic_aug = m_aug_cropped_resized
        # Adjust Gaussian noise std based on image range
        range_value = np.max(mask) - np.min(mask)
        scalar = random.uniform(0.1, 0.3)
        adjusted_std = scalar * range_value


        ##Optionally apply horizontal flip to both original and augmented
        if np.random.rand() < 0.5:
            elastic_mask = np.flip(elastic_mask, axis=1)
            elastic_aug = np.flip(elastic_aug, axis=1)

        ## Optionally apply vertical flip to both original and augmented
        if np.random.rand() < 0.5:
            elastic_mask = np.flip(elastic_mask, axis=0)
            elastic_aug = np.flip(elastic_aug, axis=0)

        yn_original.append(elastic_mask)
        yn_augmented.append(elastic_aug)

    return np.array(yn_augmented), np.array(yn_original)



def add_gaussian_noise(image, mean=0., std=0.01):
    """Add gaussian noise to a image."""
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, np.min(image), np.max(image))  # clip based on the original image range
    return noisy_image


from joblib import Parallel, delayed

def augment_image(image, mask, aug, thresh):
    mask_sum = 0
    it = 0
    while mask_sum <= thresh and it <= 5:
        augmented = aug(image=image, mask=mask)
        im_aug = augmented['image']
        m_aug = augmented['mask']
        mask_sum = int(np.sum(m_aug))
        it += 1
    return im_aug, m_aug

def aug_batchV1(batch_x, batch_y):
    (oh, ow) = (512, 512)
    aug = A.Compose([A.RandomSizedCrop(min_max_height=(200, 456), height=oh, width=ow, p=0.4),
                     A.VerticalFlip(p=0.5),
                     A.RandomRotate90(p=0.5),
                     A.OneOf([A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.3, p=.1),
                              A.Sharpen(p=.4),
                              A.Solarize(threshold=0.05, p=1)], p=0.3),
                     A.ColorJitter(brightness=0.7, contrast=0.4, saturation=0.4, hue=0.3, always_apply=False, p=0.4),
                     A.GaussNoise(var_limit=(0, 0.05), p=0.4)])

    thresh = 100
    results = Parallel(n_jobs=-1)(delayed(augment_image)(image, mask, aug, thresh) for image, mask in zip(batch_x, batch_y))
    xn, yn = zip(*results)

    return list(xn), list(yn)

import cv2
import os
import numpy as np
from random import randint

def resize_tile_df(df, num):
    # Function to apply OpenCV resize on each tile
    def resize_tile(tile):
        return cv2.resize(tile, (num, num))

    # Update 'tile' column with resized tiles
    df['tile'] = df['tile'].map(resize_tile)

    return df


def retile_labels(label_dir, save_dir, list_sizes, num_tiles_each_size=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for file_name in os.listdir(label_dir):
        if file_name.endswith('.png'):
            full_file_path = os.path.join(label_dir, file_name)
            img = cv2.imread(full_file_path, cv2.IMREAD_GRAYSCALE)

            h, w = img.shape
            original_size = (h, w)

            print(f"Processing image: {file_name}")
            print(f"Original image size: {original_size}")

            for size in list_sizes:
                for i in range(num_tiles_each_size):
                    # Choose a random top-left corner for the square
                    y = randint(0, h - size)
                    x = randint(0, w - size)

                    # Crop the square from the image
                    cropped_img = img[y:y+size, x:x+size]

                    # Print crop range
                    print(f"Crop range: x({x}, {x+size}) y({y}, {y+size})")

                    # Upscale the cropped image back to original size
                    upscaled_img = cv2.resize(cropped_img, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

                    # Save the upscaled image
                    save_path = os.path.join(save_dir, f"{file_name.split('.')[0]}_size{size}_upscaled_{i}.png")
                    cv2.imwrite(save_path, upscaled_img)

                    print(f"Saving upscaled image with tile size {size} as {save_path}")

def encode_label(fpath,s=512,buffer=False):
    if type(fpath) == str:
        img = Image.open(fpath)
    else:
        img = fpath
    shape = np.shape(img)
    if shape[0] != s or shape[0]!=s:
        if type(img) == np.ndarray:
            img = Image.fromarray(img)
        img = resize_with_padding(img, (s,s))
    label = np.array(img)
    del img # delete the img object from memory after use

    if len(label.shape) == 2:
        label = np.where(label == 0, 0.0, 1.0)
        en_label = np.expand_dims(label,axis=-1)
    elif len(label.shape) == 3 and label.shape[-1] == 4:
        l_4 = label[:, :, 3]
        en_label = np.where(l_4 == 0, 0.0, 1.0)
        en_label = np.expand_dims(en_label,axis=2)
    elif len(label.shape) == 3 and label.shape[-1] == 3:  # Check for RGB shape
        # Convert RGB to binary (0, 1) image
        #print('handling weird shape')
        label = np.mean(label, axis=2)  # Convert RGB to grayscale
        en_label = np.where(label == 0, 0.0, 1.0)
        en_label = np.expand_dims(en_label, axis=2)
    elif len(label.shape) == 3 and label.shape[-1] == 1:
        en_label = label
    else:
        raise Exception(f'Error Encoding Label: {np.shape(label)}')

    if buffer:
        print('Increasing Line Width on River Mask')
        kernel = np.ones((12,12), np.uint8)
        en_label = np.array(cv2.dilate(en_label, kernel, iterations=1))
        if en_label.ndim == 2: en_label= np.expand_dims(en_label,axis=-1)
    del label # delete the label object from memory after use

    return en_label

def tile_image(img: np.ndarray, tile_size: int=512, single_channel: bool=False, overlap=0) -> pd.DataFrame:
    """
  Tiles a large image into smaller images of a specified size.
  :param img: the image to be tiled
  :param tile_size: the size of the tiles in pixels
  :param single_channel: whether the image is single channel or not.
  :return: a dataframe containing the tiles and their coordinates
  """
    warnings.filterwarnings('ignore', category=FutureWarning)
    img = np.array(img)
    img = img.astype(np.uint8)
    (width, height) = np.shape(img)[:2]
    min_val = np.min(img)
    max_val = np.max(img)
    print(f'tiling images: {(width, height)}, {(img.dtype, max_val, min_val)} tile size: {tile_size}')
    df = pd.DataFrame(columns=['x', 'y', 'tile'])
    i = 0
    blanks = 0
    d = 0
    for x in range(0, width, tile_size - overlap):
        for y in range(0, height, tile_size - overlap):
            i += 1
            if i % 500 == 0:
                print(f'{i}')
            tile = img[x:x + tile_size, y:y + tile_size]
            (h, w) = np.shape(tile)[:2]
            if h != tile_size or w != tile_size:
                new_h = x - (tile_size - h)
                new_w = y - (tile_size - w)
                tile = img[new_h:new_h + tile_size, new_w:new_w + tile_size]
                (h, w) = np.shape(tile)[:2]
                assert h == tile_size
                assert w == tile_size
                d += 1
            # if np.sum(tile) == 0:
            #     blanks += 1
            #     continue
            if single_channel:
                if np.ndim(tile) != 3:
                    img_bytes = np.expand_dims(tile, axis=-1)
                else:
                    img_bytes = tile
            else:
                tile = Image.fromarray(tile)
                if tile.mode == 'L' or tile.mode == 'l':
                    tile = tile.convert('RGB')
                img_bytes = np.array(tile)
            df = pd.concat([df, pd.DataFrame([{'x': x, 'y': y, 'tile': img_bytes}])], ignore_index=True)
            del tile
    j = df.shape[0]
    print(f'total images: {j}, total reg images: {j}, blank images: {blanks}, wrong dimensions: {d}')
    if j == 0:
        print('all blanks')
        img_bytes = np.zeros(shape=(tile_size, tile_size))
        df = pd.concat([df, pd.DataFrame([{'x': x, 'y': y, 'tile': img_bytes}])], ignore_index=True)

    # Additional print statements for debugging
    unique_shapes = df['tile'].apply(lambda x: x.shape).unique()
    unique_dtypes = df['tile'].apply(lambda x: x.dtype).unique()
    min_max_values = df['tile'].apply(lambda x: (np.min(x), np.max(x))).unique()

    print(f"After resizing, the tiles in the DataFrame have shapes: {unique_shapes}")
    print(f"After resizing, the tiles in the DataFrame have data types: {unique_dtypes}")
    print(f"After resizing, the tiles in the DataFrame have min-max values: {min_max_values}")
    return df


def normalize8(I):
    mn = I.min()
    mx = I.max()
    mx -= mn
    I = (I - mn) / mx * 255
    return I.astype(np.uint8)



def tile_imageV1(img: np.ndarray,
               tile_size: int = 512,
               single_channel: bool = False,
               overlap: int = 0) -> pd.DataFrame:
    """
    Tiles a large image into smaller images of a specified size while providing options for customization.

    This function takes a large image and divides it into smaller tiles of a specified size,
    with optional overlap between tiles. It can also handle single-channel images and
    provides detailed information about the resulting tiles.

    Args:
        img (np.ndarray): The input image as a NumPy ndarray.
        tile_size (int, optional): The size of the tiles in pixels (default is 512).
        single_channel (bool, optional): Indicates whether the image is single-channel or not (default is False).
        overlap (int, optional): The overlap between tiles in pixels (default is 0).

    Returns:
        pd.DataFrame: A pandas DataFrame containing the tiles and their coordinates.

    Warnings:
        This function suppresses FutureWarnings generated by NumPy to maintain clean output.

    Raises:
        AssertionError: If any tile has dimensions different from the specified tile_size.

    Prints:
        - Information about the input image dimensions, data type, and intensity range.
        - Progress updates for tile processing, indicating the number of tiles processed.
        - Summary statistics including the total number of tiles, blank tiles, and tiles with incorrect dimensions.
        - A message if all tiles are blank, in which case a blank tile is created and included in the output.

    Example:
        To tile a color image 'input_image' with a tile size of 256 pixels and overlap of 64 pixels:
        >>> result_df = tile_image(input_image, tile_size=256, overlap=64)

    Note:
        - Ensure that 'img' is a valid NumPy ndarray representing an image.
        - 'tile_size' should be a positive integer representing the desired tile size.
        - 'single_channel' should be set to True if working with single-channel images (e.g., grayscale).
        - 'overlap' allows for overlapping tiles, useful for applications such as image stitching.

    """
    warnings.filterwarnings('ignore', category=FutureWarning)
    img = np.array(img)
    img = img.astype(np.uint8)
    (width, height) = np.shape(img)[:2]
    min_val = np.min(img)
    max_val = np.max(img)
    print(f'tiling images: {(width, height)}, {(img.dtype, max_val, min_val)} tile size: {tile_size}')
    df = pd.DataFrame(columns=['x', 'y', 'tile'])
    i = 0
    blanks = 0
    d = 0
    for x in range(0, width, tile_size - overlap):
        for y in range(0, height, tile_size - overlap):
            i += 1
            if i % 500 == 0:
                print(f'{i}')
            # tile = img[x:x + tile_size, y:y + tile_size]
            # (h, w) = np.shape(tile)[:2]
            # if h != tile_size or w != tile_size:
            #     new_h = x - (tile_size - h)
            #     new_w = y - (tile_size - w)
            #     tile = img[new_h:new_h + tile_size, new_w:new_w + tile_size]
            #     (h, w) = np.shape(tile)[:2]
            #     assert h == tile_size
            #     assert w == tile_size

            # First, determine if you're on the right boundary for width
            if (x + tile_size) > width:
                x = width - tile_size

            # Now, determine if you're on the bottom boundary for height
            if (y + tile_size) > height:
                y = height - tile_size

            # Now extract the tile with the potentially adjusted x and y values
            tile = img[x:x + tile_size, y:y + tile_size]

            # Verify the tile dimensions as a sanity check
            (h, w) = np.shape(tile)[:2]
            assert h == tile_size
            assert w == tile_size
            d += 1

            if np.sum(tile) == 0:
                blanks += 1
                continue
            if single_channel:
                if np.ndim(tile) != 3:
                    img_bytes = np.expand_dims(tile, axis=-1)
                else:
                    img_bytes = tile
            else:
                tile = Image.fromarray(tile)
                if tile.mode == 'L' or tile.mode == 'l':
                    tile = tile.convert('RGB')
                img_bytes = np.array(tile)
            df = pd.concat([df, pd.DataFrame([{'x': x, 'y': y, 'tile': img_bytes}])], ignore_index=True)
            del tile
    j = df.shape[0]
    print(f'total images: {j}, total reg images: {j}, blank images: {blanks}, wrong dimensions: {d}')
    if j == 0:
        print('all blanks')
        img_bytes = np.zeros(shape=(tile_size, tile_size))
        df = pd.concat([df, pd.DataFrame([{'x': x, 'y': y, 'tile': img_bytes}])], ignore_index=True)
    return df


import numpy as np

def evaluate_missed_coverage(pred_map, gt):
    """
    Computes the percentage of the ground truth not covered by the prediction.

    Parameters:
    pred_map (np.ndarray): 2D array representing the prediction map.
    gt (np.ndarray): 2D array representing the ground truth map.

    Returns:
    float: Percentage of the ground truth not covered by the prediction.
    """
    # Ensure the prediction and ground truth maps are boolean
    pred_map_bool = pred_map > 0
    gt_bool = gt > 0

    # Find areas where GT is present but the prediction is not
    missed_pred = np.logical_and(gt_bool, np.logical_not(pred_map_bool))
    missed_pred_sum = np.sum(missed_pred)

    # Compute the total ground truth area
    gt_sum = np.sum(gt_bool)

    # Compute the percentage of ground truth not covered by the prediction
    percent_missed_gt = (missed_pred_sum / gt_sum) * 100 if gt_sum != 0 else 0

    return percent_missed_gt

import rasterio
import numpy as np

def open_tiff(rasterorig, display_im=True):
    with rasterio.open(rasterorig) as src0:
        print('Original meta data: ', src0.meta)
        meta = src0.meta
        if meta['count'] >= 3:
            band1 = src0.read(1)
            band2 = src0.read(2)
            band3 = src0.read(3)
            print('3 band tiff')
            map_im = np.dstack((band1, band2, band3))
        elif meta['count'] == 1:
            map_im = src0.read(1)
            print('1 band tiff')
    return map_im if isinstance(map_im, np.ndarray) else None



def remove_small_objects_np(images, min_size=500, threshold=0.5):
    """
    Remove small objects from a batch of images.
    Args:
        images: numpy array of shape (n, 512, 512, 1)
        min_size: minimum size of connected components to keep
    Returns:
        numpy array of images with small objects removed
    """
    processed_images = []
    for img in images:
        img = img.squeeze()
        img_binary = np.where(img > threshold, True, False)
        img_cleaned = morphology.remove_small_objects(img_binary, min_size)
        processed_images.append(np.where(img_cleaned[..., np.newaxis], 1, 0))
    return np.array(processed_images)


def normalize_to_8bit(image_array):
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    #max_val = 15000

    normalized = (image_array - min_val) / (max_val - min_val)  # Normalize to [0, 1]
    eight_bit = (normalized * 255).astype(np.uint8)  # Scale to [0, 255]

    return eight_bit


def shape(x):
    print(np.shape(x))

def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill=0)


def crop_center(img, cropx, cropy):
    (w, h, c) = img.shape
    if c == 3:
        (y, x, c) = img.shape
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        return img[starty:starty + cropy, startx:startx + cropx, :]
    else:
        img = np.squeeze(img)
        (y, x) = img.shape
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx]



def encode_dataset(images_path, labels_path=None, coords=False, height=512, width=512):
    """
  TODO:
    1. Make it so that images coordinates are preserved
        parse from the name of the file
  """
    print('encoding image')
    cols = ['num', 'IMG_Padded', 'Label', 'predicted_network', 'X', 'Y']
    df = pd.DataFrame(columns=cols)
    print(df.head())
    print('df initalized')
    filenames_image = os.listdir(images_path)
    if labels_path is not None:
        filenames_mask = os.listdir(labels_path)
        filenames = [item for item in filenames_image if item in filenames_mask]
    else:
        filenames = filenames_image
    for filename in tqdm(filenames):
        if filename.endswith('.png'):
            name = re.findall('\\d+', filename)
            fname = name[0]
            print(f'found file name: {fname}')
            if coords:
                regex = '\\(([^\\)]+)\\)'
                p = re.compile(regex)
                result = p.search(filename)
                found = result.group(0).strip('()').split(',')
                x_cord = int(found[0])
                y_cord = int(found[-1])
            else:
                x_cord = None
                y_cord = None
            fpath_img = os.path.join(images_path, filename)
            np_img = encode_image(None, True, fpath_img)
            label = None
            if labels_path is not None:
                fpath = os.path.join(labels_path, filename)
                en_label = encode_label(fpath, height)
                label = [en_label]
            row = {'num': fname, 'IMG_Padded': np_img, 'Label': label, 'predicted_network': None, 'X': x_cord, 'Y': y_cord}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def encode_image(img, png=False, filepath=None, s=512):
    """
  if working with PNG there will be 4 channels, we need to open as a
  if working from how images should be tiled when inputting a geotiff,
    we will have a 1 chanel image

  """
    if png:
        img = np.array(Image.open(filepath).convert('RGB'))
    img = Image.fromarray(img, 'RGB')
    img = np.array(img)
    shp = np.shape(img)
    if shp[2] != 3:
        raise Exception('Image Needs To Have 3 channels')
    if shp[0] != s or shp[1] != s:
        img = resize_with_padding(img, (s, s))
    img = np.array(img)
    img = img / 255
    img = img.astype(np.float32)
    return img

def convert_rgb_to_grayscale(rgb_image):
    r, g, b = rgb_image[:,:,0], rgb_image[:,:,1], rgb_image[:,:,2]
    grayscale_image = 0.299*r + 0.587*g + 0.114*b
    return grayscale_image


def show(image):
    if np.ndim(image) == 3 and np.shape(image)[-1] == 1:
        image = np.squeeze(image, axis=-1)
    print(np.shape(image), np.min(image), np.max(image))
    plt.imshow(image)

def erase_loops(img: np.ndarray):
    """
    Given an image, this function erases loops (small closed shapes) that appear within the image.
    The function returns the processed image, and the count of loops that were erased.
    :param img: The image to process.
    :return: A tuple containing the processed image and the count of erased loops.
    """
    if np.ndim(img) == 3:
        img = np.squeeze(img)
    (hh, ww) = img.shape
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = contours[1] if len(contours) == 2 else contours[2]
    contours = contours[0] if len(contours) == 2 else contours[1]
    hierarchy = hierarchy[0]
    count = 0
    result = np.zeros_like(img)
    for component in zip(contours, hierarchy):
        cntr = component[0]
        hier = component[1]
        if (hier[3] > -1) & (hier[2] < 0):
            count += 1
            cv2.drawContours(result, [cntr], -1, (255, 255, 255), 7)
    return (np.expand_dims(result, axis=-1) / 255.0, count)

##############################################################
def downsize_array(arr: np.ndarray) -> np.ndarray:
    (rows, cols) = arr.shape
    new_rows = rows // 2
    new_cols = cols // 2
    return resize(arr, (new_rows, new_cols))

def crop_to_tile_image(image, target_size):
    (height, width) = np.shape(image)[:2]
    num_tiles_x = width // target_size
    num_tiles_y = height // target_size
    crop_size_x = num_tiles_x * target_size
    crop_size_y = num_tiles_y * target_size
    cropped_image = image[:crop_size_y, :crop_size_x]
    return cropped_image

def rotate_image(image, angle):
    image = tf.convert_to_tensor(image)
    (height, width) = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def align_meta_data(fp, save_path):
    with rasterio.open(fp) as src0:
        print('original meta data: ', src0.meta)
        meta1 = src0.meta
    with rasterio.open(save_path, 'r+') as src0:
        meta = src0.meta
        src0.transform = meta1['transform']
        src0.crs = meta1['crs']
        t = src0.crs




def line_lengths(binary_image):
    (contours, _) = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lengths = [cv2.arcLength(contour, closed=False) for contour in contours]
    return lengths

def average_line_length_tf(batch_images):
    batch_images_np = batch_images
    line_lengths_batch = [np.mean(line_lengths(img.squeeze().astype(np.uint8))) for img in batch_images_np]
    avg_line_length = np.mean(line_lengths_batch)
    return tf.constant(avg_line_length, dtype=tf.float32)



def stats(arr):
    """
    Print statistics about a numpy array.

    Parameters:
    - arr (numpy array): The input numpy array.

    Returns:
    - None
    """
    # Basic Information
    print("Array Statistics:")
    print("-" * 50)
    print(f"Number of Dimensions : {arr.ndim}")
    print(f"Shape                : {arr.shape}")
    print(f"Size                 : {arr.size}")
    print(f"Data Type            : {arr.dtype}")

    # Checking if the array has numeric data
    if np.issubdtype(arr.dtype, np.number):
        print(f"Minimum Value        : {np.min(arr)}")
        print(f"Maximum Value        : {np.max(arr)}")
        print(f"Mean Value           : {np.mean(arr)}")
        print(f"Standard Deviation   : {np.std(arr)}")
        # For complex numbers, print real and imaginary parts separately
        if np.iscomplexobj(arr):
            print(f"Real Part - Mean     : {np.mean(arr.real)}")
            print(f"Real Part - Std Dev  : {np.std(arr.real)}")
            print(f"Imaginary Part - Mean: {np.mean(arr.imag)}")
            print(f"Imaginary Part - Std Dev: {np.std(arr.imag)}")

    print("-" * 50)

#########################################################################333
def remove_small_objects_np(images, min_size=500, threshold=0.5):
    """
    Remove small objects from a batch of images.
    Args:
        images: numpy array of shape (n, 512, 512, 1)
        min_size: minimum size of connected components to keep
    Returns:
        numpy array of images with small objects removed
    """
    processed_images = []
    for img in images:
        img = img.squeeze()
        img_binary = np.where(img > threshold, True, False)
        img_cleaned = morphology.remove_small_objects(img_binary, min_size)
        processed_images.append(np.where(img_cleaned[..., np.newaxis], 1, 0))
    return np.array(processed_images)