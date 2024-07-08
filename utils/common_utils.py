# common_utils.py
# common_utils.py
from .common_imports import *
from .data_utils import *
from .model_utils import *
from .evaluation_utils import *
from .training_utils import *
from .image_utils import *
from .visualization_utils import *

####################################################
def print_examplesV1(x_val, y_val, model, dilate_im, step=None, num_examples=2):
    # Ensure x_val, y_val, and model are not None and have expected shapes
    if x_val is None or y_val is None or model is None or len(x_val) == 0 or len(y_val) == 0:
        print("Error: Invalid inputs provided.")
        return

    num_examples = min(num_examples, len(x_val), len(y_val))

    y_pred = model.predict(x_val[:num_examples])

    if step is not None:
        print(f'Step {step}:')
    else:
        print('End of epoch:')

    (fig, axs) = plt.subplots(num_examples, 5 if dilate_im else 3, figsize=(15, 3*num_examples))

    for i in range(num_examples):
        lbl = np.squeeze(y_val[i])
        img = np.squeeze(x_val[i])
        pred = np.squeeze(y_pred[i])

        if dilate_im:
            lbl_dilated = dilate_batchV1(np.expand_dims(y_val[i], axis=0))[0]
            lbl_dilated = np.squeeze(lbl_dilated)
            pred_masked = np.multiply(pred, lbl_dilated)
        else:
            lbl_dilated = None
            pred_masked = None

        axs[i, 0].imshow(img)
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Input Image')

        axs[i, 1].imshow(pred)
        axs[i, 1].axis('off')
        axs[i, 1].set_title('Predicted')

        axs[i, 2].imshow(lbl)
        axs[i, 2].axis('off')
        axs[i, 2].set_title('Ground Truth')

        if dilate_im:
            axs[i, 3].imshow(pred_masked)
            axs[i, 3].axis('off')
            axs[i, 3].set_title('Predicted Masked')

            axs[i, 4].imshow(lbl_dilated)
            axs[i, 4].axis('off')
            axs[i, 4].set_title('Ground Truth Dilated')

    plt.tight_layout()
    plt.show()

################################
def check_segmap_zeroes(seg_map):
    if np.sum(seg_map) == 0:
        raise ValueError("Noisy error: seg_map is completely zero after transformation!")


def get_examples(images_path, labels_path):
    """
    Returns tuple of (images, labels)
    Where each image is of size 100x100
    and each label is 100x100x2 (one-hot encoded, "is this pixel a stream or no?")

    :param images_path: path to DIRECTORY where images are kept
    :param labels_path: see above but for labels
    """
    images_lst = []
    for filename in os.listdir(images_path):
        if filename.endswith('.png'):
            img_with_chan = np.reshape(imageio.imread(images_path + filename), (100, 100, 1))
            images_lst.append(img_with_chan)
    labels_lst = []
    for filename in os.listdir(labels_path):
        if filename.endswith('.png'):
            labels_lst.append(iio.imread(labels_path + filename))
    encoded_labels = [np.zeros((100, 100)) for l in labels_lst]
    for i in range(len(labels_lst)):
        l = labels_lst[i]
        l_4 = l[:, :, 3]
        encoded_labels[i] = np.where(l_4 == 0, 0.0, 1.0)
    return (images_lst, encoded_labels)

def lut_display(image, display_min, display_max):
    lut = np.arange(2 ** 16, dtype='uint16')
    lut = display(lut, display_min, display_max)
    return np.take(lut, image)

from typing import List

def get_file_names(images_path: str, labels_path: str) -> List[str]:
    """
    Returns the file names of the images and labels that match in both directories and are shuffled.
    :param images_path: the directory path where the images are located
    :param labels_path: the directory path where the labels are located
    :return: a list of file names for the images and labels that match in both directories and are shuffled
    """
    filenames_image = os.listdir(images_path)
    filenames_mask = os.listdir(labels_path)
    files_tile = filenames_image
    files_mask = filenames_mask
    assert len(list(set(files_mask).difference(files_tile))) == 0
    assert len(list(set(files_tile).difference(files_mask))) == 0
    assert len(files_tile) == len(np.unique(files_tile))
    assert len(files_mask) == len(np.unique(files_mask))
    from sklearn.utils import shuffle
    filenames = [item for item in filenames_image if item in filenames_mask]
    (filenames_image, filenames_mask) = shuffle(filenames, filenames)
    (X_train_filenames, X_val_filenames, y_train, y_val) = train_test_split(filenames_image, filenames_mask, test_size=0.2, random_state=1)
    return (X_train_filenames, X_val_filenames)



def find_checkpoints(checkpoint_dir, step, h5=False):
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if h5:
            if filename.endswith('.h5'):
                try:
                    checkpoint_number = int(filename.split('_epoch_')[1].split('.h5')[0])
                    if checkpoint_number % step == 0:
                        checkpoints.append(os.path.join(checkpoint_dir, filename))
                except (IndexError, ValueError):
                    continue
        elif filename.endswith('.ckpt.index'):
            try:
                checkpoint_number = int(filename.split('-')[1].split('.ckpt')[0])
                if checkpoint_number % step == 0:
                    checkpoints.append(os.path.join(checkpoint_dir, filename[:-6]))
            except (IndexError, ValueError):
                continue
    return checkpoints


def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) detected: {gpus}")
        for gpu in gpus:
            print(f"Name: {gpu.name}, Type: {gpu.device_type}")
    else:
        print("No GPU detected. Running on CPU.")

# Example usage:
def transfer_metadata(original_tiff_path, prediction_array, output_path):
    """
    Transfer metadata frAom an original TIFF to a new prediction array and save it as a new TIFF.
    
    Args:
    original_tiff_path (str): Path to the original TIFF file.
    prediction_array (numpy.ndarray): The prediction array to be saved.
    output_path (str): Path where the new TIFF will be saved.
    
    Returns:
    None
    """
    # Open the original TIFF to get its metadata
    with rasterio.open(original_tiff_path) as src:
        # Get metadata
        metadata = src.meta.copy()
        
        # Convert prediction array to binary (0 and 1)
        binary_prediction = (prediction_array > 0.5).astype(np.uint8)
        
        # Update metadata for the new image
        metadata.update(
            dtype=rasterio.uint8,
            count=1,  # Single-band
            compress='deflate',
            predictor=2,  # Horizontal differencing
            zlevel=9,  # Highest compression level
            nodata=None  # No data value
        )
        
        # Create new TIFF with updated metadata
        with rasterio.open(output_path, 'w', **metadata) as dst:
            dst.write(binary_prediction, 1)  # Write the binary prediction array to the first band
            
        print(f"New TIFF saved with transferred metadata at: {output_path}")
        print(f"Final metadata: {metadata}")

       