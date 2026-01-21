import os, sys
sys.path.insert(0, os.path.abspath(".."))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.model_utils import compile_model, mean_iou, dice_loss
from utils.image_utils import open_tiff, normalize_to_8bit
from utils.visualization_utils import display
from utils.evaluation_utils import full_prediction_tiff
from utils.common_utils import transfer_metadata
import shutil
import wandb
import argparse
#import tifffile as tiff

# Clear Environment Variables
os.environ.pop('WANDB_API_KEY', None)

# Clear Wandb Config Directory
wandb_config_dir = os.path.expanduser("~/.config/wandb")
if os.path.exists(wandb_config_dir):
    shutil.rmtree(wandb_config_dir)

print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES'))

# Try to force TensorFlow to see the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#!nvidia-smi

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Is built with CUDA:", tf.test.is_built_with_cuda())
print("Is GPU available:", tf.test.is_gpu_available())


### Retrain
#!wandb login
#8b97ec4737051e4f1eecd8716131bacbcaba5e15

# #train_model(model: tf.keras.Model, images_path: str, labels_path: str, working_dir: str, epochs=20, batch_size: int=32, pretrained_weights: str=None, resize_shape=512, fine_tune=False)
# model = compile_model(512,512)
# images_path='data/New_Data/tiles'
# labels_path='data/New_Data/masks'
# working_dir='data/model_weights/riverNet/RiverNet_checkpoint_dir/training_ft_1'
# batch_size=12

# #train_modelV1(model, images_path, labels_path, working_dir, epochs=100, batch_size=batch_size, pretrained_weights=None, resize_shape=512, fine_tune=False)

"""
Setup Paths for Input and Output Directories
----------------------------------------------
Example command line execution:

python model_inference_jup.py file ../data/inputs/input.tif -s "worldview"

Arguments: 
1. file: path to input file tif.
2. -o: output file path. Default is 'data/outputs/'.
3. -s: satellite platform of input image. Options: 'wv','worldview','worldview2','worldview3','landsat',
'landsat8','landsat9','landsat5tm','landsat7','sentinel2'. Default is 'landsat'.
4. -b: manually chosen rgb bands for input image. By default, rgb bands are parsed from -s but can be manually overwritten. Enter three integers (e.g., '432') for rgb.

"""

# Implement argparser
parser = argparse.ArgumentParser(
        prog="model_inference_jup.py",
        description="A machine learning model that takes a satellite image of \
		Greenland supraglacial channels and outputs a binary map of channel locations.")
add = parser.add_argument
add("file", help="input file path")
add("-o", "--out", help="output file name/path", default="data/outputs/")
add("-s", "--sat", "--satellite", default="landsat")
add("-b", "--bands", nargs='*', type=int,
    help="manually choose rgb band numbers (overrides --satellite)")
args = parser.parse_args()
print(args.file)

# Function to create directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

# Filepath to the input TIFF file to be processed
if not os.path.splitext(args.file)[1]:
    raise ValueError("Provide an input file with an extension, not a directory.")
input_tif_fp = args.file 

# Full path where the output file will be saved
if os.path.splitext(args.out)[1]:  # file (not directory) was provided
    create_directory(os.path.dirname(args.out))  # create directory if it doesn't exist
    save_path = args.out
else:  # directory was provided
    create_directory(args.out)  # create directory if it doesn't exist
    save_path = os.path.join(args.out, os.path.basename(input_tif_fp))  # default to input name

### Trained Model Initialization
##########################################
# Snapshots of model weights were taken at epochs 70, 80, 90, and 100 to take advantage of differences in features identified at different points in training. Weighted combinations of these weights are applied to produce the final output. 
model_weights_dir = "../data/model_weights/riverNet/retrained/"
model_snapshots = [os.path.join(model_weights_dir,"model_weights_epoch_80.h5"),
                   os.path.join(model_weights_dir,"model_weights_epoch_70.h5"),
                   os.path.join(model_weights_dir,"model_weights_epoch_90.h5"),
                   os.path.join(model_weights_dir,"model_weights_epoch_100.h5")]

# Replace default tf sigmoid function that is incompatible with the current cloudspace configuration
def sigmoid(x):
    #return tf.math.sigmoid(x)
    return 1/2 + tf.keras.activations.tanh(x/2)/2

riverNet_model_list = []
for weights in model_snapshots:
   ml_model = compile_model(512,512)
   ml_model.load_weights(weights)
   riverNet_model_list.append(ml_model)
riverNet_model_list[0].get_layer('conv2d_8').activation = sigmoid
riverNet_model_list[1].get_layer('conv2d_17').activation = sigmoid
riverNet_model_list[2].get_layer('conv2d_26').activation = sigmoid
riverNet_model_list[3].get_layer('conv2d_35').activation = sigmoid

## Load seg_connector which is saved as a wandb artifact 
seg_connector = tf.keras.models.load_model(
    '../data/model_weights/segConnector/wandb_artifacts/model-training_on_own_predictions_v35',
    custom_objects={'mean_iou': mean_iou,
                    'dice_loss': dice_loss}
)

### Set Up Functions
input_ = open_tiff(input_tif_fp, args=args)
input_ = normalize_to_8bit(input_)
display(input_)
# Desired filename for the output file
#tiff.imsave('jup_input.tiff', input_)

import multiprocessing
# Manages the chunk memory efficiently for predicting on large tifs, should be able to scale to huge images
pred_map = full_prediction_tiff(input_, save_path, riverNet_model_list, seg_connector)
transfer_metadata(input_tif_fp, pred_map, save_path)

display(pred_map)

#tiff.imsave('jup_pred_map.tiff', pred_map)

import numpy as np
import matplotlib.pyplot as plt

def display_overlay(base_image, overlay_image, figsize=(20, 7)):
    """
    Display a base image, a binary overlay, and their combination in three subplots.
    
    Args:
    base_image (np.ndarray): The base image to display. Can have multiple channels.
    overlay_image (np.ndarray): The binary image to overlay. Should be 2D.
    figsize (tuple): Size of the output figure in inches. Default is (20, 7).
    
    Returns:
    None: Displays the resulting image.
    """
    # Ensure images are numpy arrays
    base_image = np.array(base_image)
    overlay_image = np.array(overlay_image)
    
    # Handle different channel configurations
    if base_image.ndim == 2:
        base_image = np.stack([base_image] * 3, axis=-1)
    elif base_image.shape[-1] not in [3, 4]:
        raise ValueError("Base image must have 1, 3, or 4 channels")
    
    if overlay_image.ndim != 2:
        raise ValueError("Overlay image must be 2D")
    
    # Create a mask for positive values
    mask = overlay_image > 0.5
    
    # Create an RGBA overlay
    overlay_rgba = np.zeros(base_image.shape[:2] + (4,))
    overlay_rgba[mask, 0] = 1  # Red for positive values
    overlay_rgba[mask, 3] = 0.5  # 50% opacity for positive values
    
    # Display the result
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Base image
    ax1.imshow(base_image[..., :3])
    ax1.set_title("Base Image")
    ax1.axis('off')
    
    # Binary overlay
    ax2.imshow(mask, cmap='binary')
    ax2.set_title("Binary Overlay")
    ax2.axis('off')
    
    # Combined overlay
    ax3.imshow(base_image[..., :3])
    ax3.imshow(overlay_rgba)
    ax3.set_title("Overlay Result")
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()


display_overlay(input_, pred_map)

