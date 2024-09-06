# evaluation_utils.py
# evaluation_utils.py
from .common_imports import *
from .common_utils import *
from .data_utils import *
from .model_utils import *
from .training_utils import *
from .image_utils import *
from .visualization_utils import *


import multiprocessing


def predict_from_tiffV1(img: np.ndarray, model: Model, fix_lines: bool=False, resize: bool=False, tile_size: int=512, overlap: int=0) -> np.ndarray:
    """
    This function takes in a tiff image and a trained model, and returns the predicted image.

    Parameters:
    img : np.ndarray - The tiff image to be predicted
    model : Model - The trained model to be used for prediction
    fix_lines : bool - A flag indicating whether to fix lines in the image (default: False)
    resize : bool - A flag indicating whether to resize the image (default: False)
    single_channel : bool - A flag indicating whether to use single channel or not (default: True)
    tile_size : int - The size of each tile to be used for prediction (default: 512)

    Returns:
    np.ndarray : The predicted image
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)
    (width_im, height_im, channels) = np.shape(img)
    print(width_im, height_im)
    if fix_lines:
        tile_df = tile_imageV1(img, tile_size, single_channel=True, overlap=overlap)
        pred_df = predict_from_dataframe_v2(tile_df, model, single_channel=True, resize=resize)
    else:
        tile_df = tile_image(img, tile_size, overlap=overlap)
        pred_df = predict_from_dataframe_v2(tile_df, model, single_channel=False, resize=resize)
    recon = reconstruct_image(pred_df, (width_im, height_im))
    del pred_df
    del tile_df
    return np.array(recon)

def ensemble_predict_from_tiffV1(img: np.ndarray, model_list: Model, fix_lines: bool=False, resize: bool=False, tile_size: int=512, overlap: int=0) -> np.ndarray:
    """
    This function takes in a tiff image and a trained model, and returns the predicted image.

    Parameters:
    img : np.ndarray - The tiff image to be predicted
    model : Model - The trained model to be used for prediction
    fix_lines : bool - A flag indicating whether to fix lines in the image (default: False)
    resize : bool - A flag indicating whether to resize the image (default: False)
    single_channel : bool - A flag indicating whether to use single channel or not (default: True)
    tile_size : int - The size of each tile to be used for prediction (default: 512)

    Returns:
    np.ndarray : The predicted image
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)
    (width_im, height_im) = np.shape(img)[:2]
    print(width_im, height_im)
    recon_fin = np.zeros(shape=np.shape(img)[:2], dtype=np.float64)
    if fix_lines:
        single_channel = True
    else:
        single_channel = False
    tile_df = tile_imageV1(img, tile_size, single_channel=single_channel, overlap=overlap)
    num_model = len(model_list)
    scaling_factors = np.linspace(0, 1, num_model)
    for (i, model) in enumerate(model_list):
        print(f'MODEL PREDICTION: {i}')
        pred_df = predict_from_dataframe_v2(tile_df, model, single_channel=single_channel, resize=resize)
        recon = reconstruct_image(pred_df, (width_im, height_im))
        recon = recon * scaling_factors[i]
        recon_fin += recon.astype(np.float64)
    del pred_df
    del tile_df
    return np.array(recon_fin / np.max(recon_fin))


####################################################################3


def parallel_ensemble_predict_from_tiff(img: np.ndarray, model_list: Model, fix_lines: bool=False, resize: bool=False, tile_size: int=512, overlap: int=0) -> np.ndarray:
    """
    This function takes in a tiff image and a trained model, and returns the predicted image.

    Parameters:
    img : np.ndarray - The tiff image to be predicted
    model : Model - The trained model to be used for prediction
    fix_lines : bool - A flag indicating whether to fix lines in the image (default: False)
    resize : bool - A flag indicating whether to resize the image (default: False)
    single_channel : bool - A flag indicating whether to use single channel or not (default: True)
    tile_size : int - The size of each tile to be used for prediction (default: 512)

    Returns:
    np.ndarray : The predicted image
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)
    (width_im, height_im) = np.shape(img)
    print(width_im, height_im)
    recon_fin = np.zeros_like(img, dtype=np.float64)
    if fix_lines:
        single_channel = True
    else:
        single_channel = False
    tile_df = tile_image(img, tile_size, single_channel=single_channel, overlap=overlap)
    num_model = len(model_list)
    scaling_factors = np.linspace(0, 1, num_model)
    for (i, model) in enumerate(model_list):
        print(f'MODEL PREDICTION: {i}')
        pred_df = predict_from_dataframe_v2(tile_df, model, single_channel=single_channel, resize=resize)
        recon = reconstruct_image(pred_df, (width_im, height_im))
        recon = recon * scaling_factors[i]
        recon_fin += recon.astype(np.float64)
    del pred_df
    del tile_df
    return np.array(recon_fin / np.max(recon_fin))
##############################################################################
def predict_checkpoints(img, model, checkpoint_dir, step):
    display(img)
    checkpoints = find_checkpoints(checkpoint_dir, step)
    p_maps = []
    for c in checkpoints:
        model.load_weights(c)
        pred_map = predict_from_tiffV1(img, model, fix_lines=False)
        binary_image = np.where(pred_map > 0.2, 1, 0)
        p_maps.append(binary_image)
    return p_maps
############################################################################
def reconstruct_image(df: pd.DataFrame, shape) -> Image:
    """
  Reconstructs an image from its tiled version using data from a dataframe
  :param df: DataFrame containing the tiled image information (x,y,tile)
  :return: Reconstructed image as a PIL Image object
  """
    row = df.iloc[0]
    (w, h, c) = np.shape(row['tile'])
    width = shape[0]
    height = shape[1]
    img = np.zeros(shape=(width, height))
    for (_, row) in tqdm(df.iterrows()):
        try:
            tile = Image.open(BytesIO(row['tile']))
            tile = np.array(tile)
        except:
            tile = np.array(row['tile'])
        x = row['x']
        y = row['y']
        tile = np.array(tile)
        if tile.ndim == 3:
            tile = np.squeeze(tile)
        (tx, ty) = np.shape(img[x:x + w, y:y + h])
        if tx != w or ty != h:
            tile = tile[:tx, :ty]
        img[x:x + w, y:y + h] += tile
        img[x:x + w, y:y + h] = np.clip(img[x:x + w, y:y + h], 0.0, 1.0)
    return img
#########################################################################
def check_segmap_zeroes(seg_map):
    if np.sum(seg_map) == 0:
        raise ValueError("Noisy error: seg_map is completely zero after transformation!")

#########################################################################

from skimage.transform import resize
import numpy as np

def ensure_minimum_size(image, min_size=512):
    """
    Upscales an image to ensure its smallest dimension is at least the specified minimum size.

    Args:
    image (numpy.ndarray): Input image array.
    min_size (int): Minimum size for the smallest dimension (default: 512).

    Returns:
    numpy.ndarray: Resized image with the smallest dimension at least 'min_size' pixels.
                   If the input image is already large enough, it is returned unchanged.
    """
    # Get current image dimensions
    height, width = image.shape[:2]
    
    # Check if resizing is necessary
    if height >= min_size and width >= min_size:
        return image
    
    # Calculate scaling factor
    scale = max(min_size / height, min_size / width)
    
    # Calculate new dimensions
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    # Resize the image
    resized_image = resize(image, (new_height, new_width), 
                           anti_aliasing=True, 
                           preserve_range=True)
    
    # Ensure the output has the same data type as the input
    resized_image = resized_image.astype(image.dtype)
    
    print(f"Image resized from {(height, width)} to {(new_height, new_width)}")
    
    return resized_image

def revert_to_original_size(resized_image, original_shape):
    """
    Resizes an image back to its original dimensions if it was previously upscaled.

    Args:
    resized_image (numpy.ndarray): The potentially resized image array.
    original_shape (tuple): The original shape of the image (height, width).

    Returns:
    numpy.ndarray: Image resized to the original dimensions if it was upscaled,
                   or the input image if no resizing is needed.
    """
    current_height, current_width = resized_image.shape[:2]
    original_height, original_width = original_shape[:2]

    # Check if resizing is necessary
    if (current_height, current_width) == (original_height, original_width):
        return resized_image

    # Resize the image back to original dimensions
    original_image = resize(resized_image, (original_height, original_width), 
                            anti_aliasing=True, 
                            preserve_range=True)

    # Ensure the output has the same data type as the input
    original_image = original_image.astype(resized_image.dtype)

    print(f"Image resized from {(current_height, current_width)} back to original size {(original_height, original_width)}")

    return original_image


def full_prediction_tiff(map, save_path, RiverNet_list, seg_conncector):
    """
  Arguments:

  map: a 2D array representing an image to be segmented
  save_path: a filepath to the directory to which the final tiff will be saved
  model_list: a list of ensemble learners to be run on satelliete data
  model_lines: morphological neural network to be run over output of model_list

  Summary:
  The full_prediction_tiff function performs segmentation on an input image by
  breaking it down into smaller chunks and applying a pre-trained segmentation
  model to each chunk. It then reassembles the predicted segmentation maps for each
  chunk into a single output segmentation map for the entire image. The function
  uses a pre-defined checkpoint directory containing pre-trained models to perform
  the segmentation. It prints the dimensions of the input image before and after
  cropping, as well as the total number of chunks to be processed. The function
  returns the output segmentation map.

  """
    original_shape = np.shape(map)

    image = ensure_minimum_size(map)
    chunk_size = 512 * 10
    print(f'Image Before Crop: {np.shape(image)}')
    print(f'Image After Crop: {np.shape(image)}')
    image_height = image.shape[0]
    image_width = image.shape[1]
    total_chunks = image_height * image_width / (chunk_size * chunk_size)
    print(f'TOTAL CHUNKS TO BE PROCESSED: {total_chunks}')
    pred_map_full = np.zeros((image_height, image_width))
    print('pred_map intialized')
    cur_chunk = 0
    for i in range(0, image_height, chunk_size):
        for j in range(0, image_width, chunk_size):
            print(f'*****************************************************************')
            print(f'*****************CHUNKS PROCESSED: {cur_chunk}*****************')
            i_min = i
            i_max = min(i + chunk_size, image_height)
            j_min = j
            j_max = min(j + chunk_size, image_width)

            if i_max == image_height:  # this is the last row of chunks
                i_min = image_height - chunk_size
                i_max = image_height

            if j_max == image_width:  # this is the last column of chunks
                j_min = image_width - chunk_size
                j_max = image_width

            chunk = image[i_min:i_max, j_min:j_max]
            shape(chunk)
            #seg_map = np.zeros(shape=(chunk_size,chunk_size))
            seg_map = ensemble_predict_from_tiffV1(chunk, RiverNet_list, resize=False, fix_lines=False, overlap=50)
            print(' ********************* Finished Ensemble Prediction with RiverNet for Chunk ********************')
            seg_map = np.where(seg_map > 0.1, 1, 0)
            check_segmap_zeroes(seg_map)

            if seg_map.ndim == 2: seg_map = np.expand_dims(seg_map, axis=-1)
            print(f'seg_map: {np.shape(seg_map)}, pred_map: {np.shape(pred_map_full[i_min:i_max, j_min:j_max])}')

            #One Pass At Low Resolution
            seg_map = predict_from_tiffV1(seg_map, seg_conncector, fix_lines=True, resize=False, tile_size = 512, overlap=150)
            seg_map = np.where(seg_map > 0.1, 1, 0)
            if seg_map.ndim == 2: seg_map = np.expand_dims(seg_map, axis=-1)
            seg_map = predict_from_tiffV1(seg_map, seg_conncector, fix_lines=True, resize=False, tile_size = 512, overlap=150)

            seg_map = np.where(seg_map > 0.1, 1, 0)
            if seg_map.ndim == 2: seg_map = np.expand_dims(seg_map, axis=-1)
            seg_map = predict_from_tiffV1(seg_map, seg_conncector, fix_lines=True, resize=False, tile_size = 512, overlap=150)

            seg_map = np.where(seg_map > 0.1, 1, 0)
            if seg_map.ndim == 2: seg_map = np.expand_dims(seg_map, axis=-1)
            seg_map = predict_from_tiffV1(seg_map, seg_conncector, fix_lines=True, resize=False, tile_size = 512, overlap=150)

            seg_map = np.where(seg_map > 0.1, 1, 0)
            if seg_map.ndim == 2: seg_map = np.expand_dims(seg_map, axis=-1)
            seg_map = predict_from_tiffV1(seg_map, seg_conncector, fix_lines=True, resize=False, tile_size = 512, overlap=150)

            check_segmap_zeroes(seg_map)

            # for i in range(6):
            #   offset=50
            #   print(f'^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            #   print(f'SegConnector Prediction: {i}')
            #   if seg_map.ndim == 2: seg_map = np.expand_dims(seg_map, axis=-1)
            #   seg_map = predict_from_tiffV1(seg_map, seg_conncector, fix_lines=True, resize=False, tile_size = 512, overlap=50+offset*i)
            #   check_segmap_zeroes(seg_map)

            ##Higher Overlap
            #seg_map = predict_from_tiff(seg_map, seg_conncector, fix_lines=True, resize=False, tile_size = 512, overlap=150)

            #####################################3
            #if seg_map.ndim == 2: seg_map = np.expand_dims(seg_map, axis=-1)

            # print(f"Original shape: {np.shape(seg_map)}")  # Debug print
            # original_shape = seg_map.shape  # Storing original shape for later

            # # Downscale seg_map by 2 on each dimension using scikit-image
            # seg_map_downscaled = resize(seg_map, (original_shape[0]//2, original_shape[1]//2, original_shape[2]), anti_aliasing=True)
            # print(f"Shape after downscaling: {np.shape(seg_map_downscaled)}")  # Debug print

            # # Make prediction on the downscaled image
            # seg_map_downscaled = predict_from_tiff(seg_map_downscaled, seg_connector, fix_lines=True, resize=False, tile_size=512, overlap=50)
            # print(f"Shape after prediction: {np.shape(seg_map_downscaled)}")  # Debug print

            # if seg_map_downscaled.ndim == 2: seg_map_downscaled = np.expand_dims(seg_map_downscaled, axis=-1)

            # # Upscale seg_map back to its original size using scikit-image
            # seg_map = np.squeeze(resize(seg_map_downscaled, original_shape, anti_aliasing=True))

            # print(f"Shape after upscaling: {np.shape(seg_map)}")  # Debug print

            # # Make sure the rescaled shape is the same as the original shape
            # #assert seg_map.shape == original_shape, "Shapes are not equal"


            #################################
            # Adjusting for overlap on the boundary
            print(f'seg_map: {np.shape(seg_map)}, pred_map: {np.shape(pred_map_full[i_min:i_max, j_min:j_max])}')
            if np.shape(seg_map) == np.shape(pred_map_full[i_min:i_max, j_min:j_max]):
                print('correct dimensions pasting')
                pred_map_full[i_min:i_max, j_min:j_max] = seg_map
            else:
                print('incorrect dimensions pasting')
                (h, w) = np.shape(pred_map_full[i_min:i_max, j_min:j_max])
                pred_map_full[i_min:i_max, j_min:j_max] = seg_map[:h, :w]
            del chunk
            del seg_map
            cur_chunk += 1


            # chunk_border_thickness = 30  # You can adjust this if you want thicker borders
            # pred_map_full[i_min:i_min+chunk_border_thickness, j_min:j_max] = 1
            # pred_map_full[i_max-chunk_border_thickness:i_max, j_min:j_max] = 1
            # pred_map_full[i_min:i_max, j_min:j_min+chunk_border_thickness] = 1
            # pred_map_full[i_min:i_max, j_max-chunk_border_thickness:j_max] = 1
    pred_map_full = revert_to_original_size(pred_map_full, original_shape)

    try:
        mask = (map == 0)
        pred_map_full = pred_map_full * ~mask

    except:
        mask = (map[:,:,0] == 0)
        pred_map_full = pred_map_full * ~mask

    if save_path is not None:
        try:
            tifffile.imsave(save_path, pred_map_full)
        except:
            print('saving failed')
    print('Retreiving Final Prediction')
    print(f'Final size: {np.shape(pred_map_full)}')
    return pred_map_full


def optimized_full_prediction_tiff(map, save_path, RiverNet_list, seg_connector, overlap=50):
    image = map
    chunk_size = 512 * 10
    image_height, image_width = image.shape[:2]
    
    def get_tile_bounds(i, j):
        i_min, i_max = max(0, i-overlap), min(image_height, i+chunk_size+overlap)
        j_min, j_max = max(0, j-overlap), min(image_width, j+chunk_size+overlap)
        return i_min, i_max, j_min, j_max

    tiles = [image[get_tile_bounds(i, j)] 
             for i in range(0, image_height, chunk_size) 
             for j in range(0, image_width, chunk_size)]
    
    def process_chunk(chunk):
        seg_map = np.zeros_like(chunk, dtype=np.float64)
        for i, model in enumerate(RiverNet_list):
            pred = predict_from_tiffV1(chunk, model, resize=False, fix_lines=False, overlap=50)
            seg_map += pred * (i + 1) / len(RiverNet_list)
        
        seg_map = np.where(seg_map > 0.1, 1, 0)
        check_segmap_zeroes(seg_map)
        
        for _ in range(5):
            if seg_map.ndim == 2: 
                seg_map = np.expand_dims(seg_map, axis=-1)
            seg_map = predict_from_tiffV1(seg_map, seg_connector, fix_lines=True, resize=False, tile_size=512, overlap=150)
            seg_map = np.where(seg_map > 0.1, 1, 0)
            check_segmap_zeroes(seg_map)
        
        return seg_map

    print("Processing chunks...")
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(process_chunk, tiles), total=len(tiles)))
    
    print("Reconstructing image...")
    pred_map_full = np.zeros((image_height, image_width))
    idx = 0
    for i in range(0, image_height, chunk_size):
        for j in range(0, image_width, chunk_size):
            i_min, i_max, j_min, j_max = get_tile_bounds(i, j)
            i_start, j_start = i - i_min, j - j_min
            i_end, j_end = i_start + chunk_size, j_start + chunk_size
            pred_map_full[i:min(i+chunk_size, image_height), j:min(j+chunk_size, image_width)] = \
                results[idx][i_start:i_end, j_start:j_end]
            idx += 1
    
    if save_path is not None:
        try:
            print(f"Saving result to {save_path}")
            tifffile.imwrite(save_path, pred_map_full)
            print("Save successful")
        except Exception as e:
            print(f"Saving failed: {str(e)}")
    
    return pred_map_full

###########################
def predict_from_dataframe(image_df, model):
    """
    Input: image_df
    Output: image_df with the imaged prediction map for each tile

    each image within image_df is an unprocessed numpy array (100x100)
      psuedo code:
        for each image in image_df
          process the image into 128,128,3
          make a prediction on the image
          save the prediction into
   cols = ['num', 'IMG_Padded', 'Label', 'predicted_network', 'X', 'Y']

  """
    for (index, row) in df.iterrows():
        each_image = row['IMG_Padded']
        if index % 50 == 0:
            print(f'Images Predicted: {index}')
        img_processed = each_image
        prediction_map = model.predict(np.expand_dims(img_processed, axis=0), verbose=0)
        pred_map = prediction_map.squeeze(axis=0).squeeze(axis=2)
        row['predicted_network'] = pred_map
    return image_df

##############################3



def predict_from_dataframe_v2(df: pd.DataFrame, model, batch_size: int=64, resize: bool=False, single_channel: bool=False) -> pd.DataFrame:
    """
    This function predicts on image tiles stored in a dataframe using a pre-trained model
    df : pd.DataFrame : dataframe containing image tiles
    model : keras.model : pre-trained model
    batch_size : int : number of images to be predicted at once
    resize : bool : whether to resize the images
    single_channel : bool : whether the images are single channel or not
    """
    if single_channel:
        print('single channel tiff')
        images = df['tile'].apply(lambda x: encode_label(x)).tolist()
    else:
        print('3 channel tiff')
        images = df['tile'].apply(lambda x: encode_image(x)).tolist()
    print('Images From Dataframe Are Properly Encoded')
    res = int(model.input.shape[1])
    if resize:
        print('resizing images to model input')
        images = tf.image.resize(images, [res, res])
        print('Resizing Before Prediction: ', np.shape(images))
    print('images are proccesed: ', np.shape(images))
    images = np.stack(images, axis=0)
    assert images.ndim == 4
    image = images[0]
    print('tiled image data: ', np.shape(images), np.min(image), np.max(image), image.dtype)
    predictions = model.predict(images, verbose=1)
    gc.collect()
    print('prediction data shape: ', np.shape(predictions))
    pred = []
    for im in predictions:
        pred.append([im])
    del predictions
    ex_df = pd.DataFrame(data=pred, columns=['tile'])
    ex_df['x'] = df['x']
    ex_df['y'] = df['y']
    del df
    return ex_df

def apply_function_to_tiff(img: np.ndarray, func: callable, single_channel: bool, resize: bool=False, tile_size: int=512):
    """
  Applies a function to a tiff image and returns the processed image.
  :param img: 2D tiff (height, width)
  :param func: a function that takes in a 2/3 channel image and outputs a 3 channel image
  :param single_channel: a flag indicating if the input image is single channel
  :param resize: a flag indicating if the image should be resized before processing
  :param tile_size: the size of the tiles to split the image into
  :return: the processed image
  """
    assert np.ndim(img) == 2
    (width, height) = np.shape(img)[0:2]
    print(width, height)
    if single_channel:
        tile_df = tile_image(img, tile_size, single_channel=True)
    else:
        tile_df = tile_image(img, tile_size)
    pred = []
    for im in tile_df['tile']:
        pred.append([func(im)])
    print(f'recon preds: {np.shape(pred)},raw preds: {np.shape(pred)}')
    ex_df = pd.DataFrame(data=pred, columns=['tile'])
    ex_df['x'] = tile_df['x']
    ex_df['y'] = tile_df['y']
    recon = reconstruct_image(ex_df)
    return np.array(recon)
#################################
def full_prediction_tiff_single_model(map, save_path, model, input_tif_fp=None):
    """
  Arguments:

  map: a multi-band array representing an image to be segmented
  save_path: a filepath to the directory to which the final tiff will be saved
  model_list: a list of ensemble learners to be run on satelliete data
  model_lines: morphological neural network to be run over output of model_list

  Summary:
  The full_prediction_tiff function performs segmentation on an input image by
  breaking it down into smaller chunks and applying a pre-trained segmentation
  model to each chunk. It then reassembles the predicted segmentation maps for each
  chunk into a single output segmentation map for the entire image. The function
  uses a pre-defined checkpoint directory containing pre-trained models to perform
  the segmentation. It prints the dimensions of the input image before and after
  cropping, as well as the total number of chunks to be processed. The function
  returns the output segmentation map.

  """
    image = map
    chunk_size = 512 * 10
    print(f'Image Before Crop: {np.shape(image)}')
    print(f'Image After Crop: {np.shape(image)}')
    image_height = image.shape[0]
    image_width = image.shape[1]
    total_chunks = image_height * image_width / (chunk_size * chunk_size)
    print(f'TOTAL CHUNKS TO BE PROCESSED: {total_chunks}')
    pred_map_full = np.zeros((image_height, image_width))
    print('pred_map intialized')
    cur_chunk = 0



    for i in range(0, image_height, chunk_size):
        for j in range(0, image_width, chunk_size):
            print(f'*****************CHUNKS PROCESSED: {cur_chunk}*****************')
            i_min = i
            i_max = min(i + chunk_size, image_height)
            j_min = j
            j_max = min(j + chunk_size, image_width)
            chunk = image[i_min:i_max, j_min:j_max]
            shape(chunk)
            #chunk = np.expand_dims(chunk,axis=-1)
            shape(chunk)

            try:

              seg_map = predict_from_tiffV1(chunk, model, resize = True, tile_size =512, fix_lines=True, overlap=20)

            except Exception as e:
              print(f"An error occurred while predicting from TIFF: {e}")
              seg_map = np.zeros_like(chunk)  # Filling with zeros

            print(f'Unique Values in seg_map prediction: {np.unique(seg_map)}')
            print(f'seg_map: {np.shape(seg_map)}, pred_map: {np.shape(pred_map_full[i_min:i_max, j_min:j_max])}')

            #seg_map = np.where(seg_map > 0.1, 1, 0)
            #seg_map = predict_from_tiff(seg_map, model_lines, fix_lines=True, resize=False, overlap=50)
            print(f'seg_map: {np.shape(seg_map)}, pred_map: {np.shape(pred_map_full[i_min:i_max, j_min:j_max])}')
            if np.shape(seg_map) == np.shape(pred_map_full[i_min:i_max, j_min:j_max]):
                print('correct dimensions pasting')
                pred_map_full[i_min:i_max, j_min:j_max] = seg_map
            else:
                print('incorrect dimensions pasting')
                (h, w) = np.shape(pred_map_full[i_min:i_max, j_min:j_max])
                pred_map_full[i_min:i_max, j_min:j_max] = seg_map[:h, :w]
            del chunk
            del seg_map
            cur_chunk += 1


    try:
        pred_map_full = mask_nan(pred_map_full,map)
    except:
        print('masking failed failed')

    if save_path is not None and input_tif_fp is not None:
        try:
            transfer_metadata(input_tif_fp, pred_map, save_path)    
        except:
            print('saving failed')


    print('Retreiving Final Prediction')
    return pred_map_full

###############################################################3


def mask_nan(pred_map,m):
  try:
    mask = (m == 0)
    pred_map_full = pred_map * ~mask

  except:
    mask = (m[:,:,0] == 0)
    pred_map_full = pred_map * ~mask

  return pred_map_full