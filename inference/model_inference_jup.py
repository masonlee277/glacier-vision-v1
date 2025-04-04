import os, sys
sys.path.insert(0, os.path.abspath(".."))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
import shutil
import wandb

import tifffile as tiff

# Step 1: Clear Environment Variables
os.environ.pop('WANDB_API_KEY', None)

# Step 2: Clear Wandb Config Directory
wandb_config_dir = os.path.expanduser("~/.config/wandb")
if os.path.exists(wandb_config_dir):
    shutil.rmtree(wandb_config_dir)

import os
print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES'))

# Try to force TensorFlow to see the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#!nvidia-smi

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Is built with CUDA:", tf.test.is_built_with_cuda())
print("Is GPU available:", tf.test.is_gpu_available())


### Train
#!wandb login
#8b97ec4737051e4f1eecd8716131bacbcaba5e15

# ### this is how to

# #train_model(model: tf.keras.Model, images_path: str, labels_path: str, working_dir: str, epochs=20, batch_size: int=32, pretrained_weights: str=None, resize_shape=512, fine_tune=False)
# model = compile_model(512,512)
# images_path='data/New_Data/tiles'
# labels_path='data/New_Data/masks'
# working_dir='data/model_weights/riverNet/RiverNet_checkpoint_dir/training_ft_1'
# batch_size=12

# #train_modelV1(model, images_path, labels_path, working_dir, epochs=100, batch_size=batch_size, pretrained_weights=None, resize_shape=512, fine_tune=False)

### Running Seg Connector
## Define file paths
"""
Setup Paths for Input and Output Directories
----------------------------------------------

In this section, we configure various paths used by our program. These paths are to the input, output, and model weights directories, and to the input TIFF file. We also specify the filename for the desired output file.

Please make sure to replace these paths with the correct paths for your own project.

Here is the purpose of each path:

1. path: This is the root path where your project is located.
2. output_dir: This is the path where you want to save your output files.
3. input_dir: This is the path where your input files are located.
4. model_weights_dir: This is the path where your model weights are located.
5. input_tif_fp: This is the filepath to the input TIFF file that you want to process.
6. desired_output_filename: This is the filename that you want to give to your output file.
7. save_path: This is the full path where your output file will be saved.

google bucket link: https://console.cloud.google.com/storage/browser/greenland_delin_imagery;tab=objects?prefix&forceOnObjectsSortingFiltering=false&pli=1
"""

import os
import sys
print(sys.version)

# Path to the root directory of the project
path = 'data/outputs'

# Path to the output directory where the results will be saved
output_dir = os.path.join(path, 'outputs')

# Path to the input directory where the input files are located
input_dir = os.path.join(path, 'inputs')

# Filepath to the input TIFF file to be processed
## Load the tif and preprocess for the model
#input_tif_fp = 'data/sat_images/neiv-validation-data/WV03_20220801143842_1040010079411F00_22AUG01143842-M1BS-506796344080_01_P001_u16rf3413_RGB_COMP_CROPPED.tif'
input_tif_fp = '../data/mark_validation/clip_LC09_L2SP_006013_20220728_20230406_02_T1_RGB_COMP_cropped.tif'
#input_tif_fp = '/teamspace/studios/this_studio/data/mark_validation/T21XVK_20200817T193911_truecolor_clipped.tif'
#input_tif_fp =  '/teamspace/studios/this_studio/data/mark_validation/clip_T22WEV_20220801T150809_RGB_COMP_10m_CROPPED.tif'
#input_tif_fp = 'data/mark_validation/clip_WV03_20220801143842_1040010079411F00_22AUG01143842-M1BS-506796344080_01_P001_u16rf3413_RGB_COMP_CROPPED.tif'
desired_output_filename = '../data/mark_validation/jup_output_T21XVK_20200817T193911_truecolor_clipped.tif'

# Full path where the output file will be saved
#save_path = os.path.join(output_dir, desired_output_filename)
save_path = desired_output_filename

# Function to create directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

# Create directories if they don't exist
create_directory(path)
create_directory(output_dir)
create_directory(input_dir)

### Trained Model Initialization
##########################################
##using a single ml_model
# ml_model = compile_model(512,512)
# c =  "/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/DB512v6/IOU/checkpoint_dir/cp-0008.ckpt"
# ml_model.load_weights(c)
#!ls 'data/model_weights/riverNet/RiverNet_checkpoint_dir/retiled_dice_loss_A100_no_aug-10-9-223'

##########################################
#model_weights_dir = "/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/New_Data/training_dir/RiverNet_checkpoint_dir/retiled_dice_loss_A100_no_aug-10-9-223"
#model_weights_dir = "data/model_weights/riverNet/RiverNet_checkpoint_dir/retrained"
#model_weights_dir = "/teamspace/studios/this_studio/data/model_weights/riverNet/retrained/"
model_weights_dir = "../data/model_weights/riverNet/retrained/"
#ch = find_checkpoints(model_weights_dir,2)
# ch= [os.path.join(model_weights_dir,"model_weights_epoch_4.h5"),
#      os.path.join(model_weights_dir,"model_weights_epoch_12.h5"),
#      os.path.join(model_weights_dir,"model_weights_epoch_20.h5"),
#      os.path.join(model_weights_dir,"model_weights_epoch_28.h5")]

ch= [os.path.join(model_weights_dir,"model_weights_epoch_80.h5"),
     os.path.join(model_weights_dir,"model_weights_epoch_70.h5"),
     os.path.join(model_weights_dir,"model_weights_epoch_90.h5"),
     os.path.join(model_weights_dir,"model_weights_epoch_100.h5")]

import pdb
def activation_db(x):
    print("!!!!sigmoid!!!!")
    print(x)
    print("....")
    #return tf.math.sigmoid(*args, **kwargs)
    return tf.keras.activations.relu(x)

riverNet_model_list = []
for c in ch:
   print(c) #all the epochs of the checkpoints
   ml_model = compile_model(512,512)
   ml_model.load_weights(c)
   riverNet_model_list.append(ml_model)
riverNet_model_list[0].get_layer('conv2d_8').activation = activation_db
riverNet_model_list[1].get_layer('conv2d_17').activation = activation_db
riverNet_model_list[2].get_layer('conv2d_26').activation = activation_db
riverNet_model_list[3].get_layer('conv2d_35').activation = activation_db

print(riverNet_model_list[0].summary())
print(riverNet_model_list[1].summary())
print(riverNet_model_list[2].summary())
print(riverNet_model_list[3].summary())


##########################################

# import wandb
# import wandb
# wandb.api.clear_setting('api_key')
# # Force re-login
# wandb.login(relogin=True)
# # Step 1: Log out of the current session
# import wandb

# run = wandb.init()
# artifact = run.use_artifact('northern-change/segconnectorv2/model-training_on_RiverNet_PredictionsV2:v29', type='model')
# artifact_dir = artifact.download()




## Load seg_connector which is saved as a wandb artifact 
seg_connector = tf.keras.models.load_model(
    #'data/model_weights/segConnector/wandb_artifacts/model-training_on_RiverNet_PredictionsV2:v29',
    '../data/model_weights/segConnector/wandb_artifacts/model-training_on_own_predictions_v35',
    custom_objects={'mean_iou': mean_iou,
                    'dice_loss': dice_lossV1}
)


seg_connector.get_layer('conv2d_80').activation = activation_db
## Didn't import other wandb set up from lightningai

### Set Up Functions
input = open_tiff(input_tif_fp)
input = normalize_to_8bit(input)
display(input)# Desired filename for the output file
tiff.imsave('jup_input.tiff', input)


from utils import *
import multiprocessing
#Manages the chunk memory efficiently for predicting on large tifs, should be able to scale to huge images
print(seg_connector.summary())
pred_map = full_prediction_tiff(input, save_path, riverNet_model_list, seg_connector)
transfer_metadata(input_tif_fp, pred_map, "./jup_output_test.tiff")

input.shape

display(pred_map)

tiff.imsave('jup_pred_map.tiff', pred_map)

pred_map.shape

np.unique(pred_map)

input.shape

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


display_overlay(input, pred_map)


### Prediction Bucket Data 
# Didn't import because not using Google Buckets

'''
### Prediction Individual Tifs
#!ls "/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/New_Data/for_mason/need_buffering"

#input_tif_fp = "/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/New_Data/for_mason/need_buffering/sn2_VIS.tif"
input = open_tiff(input_tif_fp,display_im=False)
input = normalize_to_8bit(input)
with rasterio.open(input_tif_fp) as src:
    original_meta = src.meta
    print(original_meta)
stats(input)

display(input[::20, ::20]) ## downscale

save_path = None
print(input.shape)
pred_map = full_prediction_tiff(input, save_path, model_list, seg_connector)
print(pred_map.shape)
try:
  mask = (input == 0)
  pred_map = pred_map * ~mask

except:
  mask = (input[:,:,0] == 0)
  pred_map = pred_map * ~mask

pred_map = pred_map.astype(np.uint8) # compress
stats(pred_map)

display(pred_map)
#download_tiff(pred_map,original_meta, filename='sn2_pred.tif')

display(pred_map[4000:7000, 4000:7000])

import os
def count_files_in_directory(directory_path):
    with os.scandir(directory_path) as entries:
        return sum(1 for entry in entries if entry.is_file())

directory_path = "/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/New_Data/seg_connector_tiles/PredV3/mask"
file_count = count_files_in_directory(directory_path)
print(f"Number of files in directory: {file_count}")

gt_tif_fp = "/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/New_Data/for_mason/need_buffering/sn2_gt.tif"
gt = open_tiff(input_tif_fp,display_im=False)
gt = normalize_to_8bit(gt)

display(gt)
'''
