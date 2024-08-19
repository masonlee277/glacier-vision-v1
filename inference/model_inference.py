#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *
import os
import shutil
import wandb

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

get_ipython().system('nvidia-smi')

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Is built with CUDA:", tf.test.is_built_with_cuda())
print("Is GPU available:", tf.test.is_gpu_available())


# # Train

# In[2]:


#!wandb login
#8b97ec4737051e4f1eecd8716131bacbcaba5e15


# In[3]:


# ### this is how to

# #train_model(model: tf.keras.Model, images_path: str, labels_path: str, working_dir: str, epochs=20, batch_size: int=32, pretrained_weights: str=None, resize_shape=512, fine_tune=False)
# model = compile_model(512,512)
# images_path='data/New_Data/tiles'
# labels_path='data/New_Data/masks'
# working_dir='data/model_weights/riverNet/RiverNet_checkpoint_dir/training_ft_1'
# batch_size=12

# #train_modelV1(model, images_path, labels_path, working_dir, epochs=100, batch_size=batch_size, pretrained_weights=None, resize_shape=512, fine_tune=False)


# # Running Seg Connector

# ## Define File Paths

# In[4]:


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
input_tif_fp = 'data/sat_images/neiv-validation-data/WV03_20220801143842_1040010079411F00_22AUG01143842-M1BS-506796344080_01_P001_u16rf3413_RGB_COMP_CROPPED.tif'

desired_output_filename = 'output_test_3.tif'

# Full path where the output file will be saved
save_path = os.path.join(output_dir, desired_output_filename)


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



# ## Trained Model Intializaiton

# In[5]:


##########################################
##using a single ml_model
# ml_model = compile_model(512,512)
# c =  "/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/DB512v6/IOU/checkpoint_dir/cp-0008.ckpt"
# ml_model.load_weights(c)
get_ipython().system("ls 'data/model_weights/riverNet/RiverNet_checkpoint_dir/retiled_dice_loss_A100_no_aug-10-9-223'")

##########################################
#model_weights_dir = "/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/New_Data/training_dir/RiverNet_checkpoint_dir/retiled_dice_loss_A100_no_aug-10-9-223"
model_weights_dir = "data/model_weights/riverNet/RiverNet_checkpoint_dir/retrained"
#ch = find_checkpoints(model_weights_dir,2)
# ch= [os.path.join(model_weights_dir,"model_weights_epoch_4.h5"),
#      os.path.join(model_weights_dir,"model_weights_epoch_12.h5"),
#      os.path.join(model_weights_dir,"model_weights_epoch_20.h5"),
#      os.path.join(model_weights_dir,"model_weights_epoch_28.h5")]

ch= [os.path.join(model_weights_dir,"model_weights_epoch_80.h5"),
     os.path.join(model_weights_dir,"model_weights_epoch_70.h5"),
     os.path.join(model_weights_dir,"model_weights_epoch_90.h5"),
     os.path.join(model_weights_dir,"model_weights_epoch_100.h5")]

riverNet_model_list = []
for c in ch:
   print(c) #all the epochs of the checkpoints
   ml_model = compile_model(512,512)
   ml_model.load_weights(c)
   riverNet_model_list.append(ml_model)


##########################################


# In[6]:


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


# In[7]:


## Load seg_connector which is saved as a wandb artifact 
seg_connector = tf.keras.models.load_model(
    'data/model_weights/segConnector/wandb_artifacts/model-training_on_RiverNet_PredictionsV2:v29',
    custom_objects={'mean_iou': mean_iou,
                    'dice_loss': dice_lossV1}
)


# ## Set Up Functions

# In[8]:


input = open_tiff(input_tif_fp)
input = normalize_to_8bit(input)
display(input)# Desired filename for the output file


# In[9]:


from utils import *
import multiprocessing
#Manages the chunk memory efficiently for predicting on large tifs, should be able to scale to huge images
pred_map = full_prediction_tiff(input, save_path, riverNet_model_list, seg_connector)
transfer_metadata(input_tif_fp, pred_map, save_path)


# In[10]:


input.shape


# In[11]:


display(pred_map)


# In[12]:


display(pred_map_full)


# In[ ]:


pred_map.shape


# In[ ]:


input.shape


# In[ ]:





# ## Prediction Bucket Data

# THIS DOES NOT WORK BECAUSE NOT IN THE GCLOUD ATM. Please import nessecary modules and figure this out if you need access here. 

# In[ ]:


# Function to download a TIFF file from a Google Cloud Storage bucket
def download_tiff_from_bucket(file_path, bucket_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    local_file_path = file_path.split('/')[-1]
    blob.download_to_filename(local_file_path)
    return local_file_path

def process_tiff_file_from_bucket(file_path, bucket_name='greenland_delin_imagery'):
    local_file_path = download_tiff_from_bucket(file_path, bucket_name)
    return open_tiff(local_file_path)


# In[ ]:


from google.colab import auth
auth.authenticate_user()

from google.cloud import storage
client = storage.Client()
bucket = client.get_bucket('greenland_delin_imagery')
blobs = bucket.list_blobs()
# for file in files:
#     print(file.name)


# Create an empty list to store file names
tif_files = []

# Iterate over each blob
for blob in blobs:
    # Check if the file is a .tif file
    if blob.name.endswith('.tif'):
        # Append the blob name to the list
        tif_files.append(blob.name)
# Print out the list of .tif files
for i in tif_files: print(i)


# In[ ]:


file_path = "pred_batch_1/WVImagery/Minturn/Partial watershed 2019/WV02_20190804171102_103001009880B200_19AUG04171102-P1BS-503581603060_01_P003_u16rf3413-pred-v2.tif"
map = process_tiff_file_from_bucket(file_path)


# In[ ]:


import numpy as np
import rasterio
import traceback
import os

from google.cloud import storage
# Instantiate a Google Cloud Storage client
client = storage.Client()

# Specify your bucket
bucket_name = 'greenland_delin_imagery'
bucket = client.get_bucket(bucket_name)

# List all the blobs in the bucket
blobs = bucket.list_blobs()
local_path = ''
tiff_filepath = 'temp_download.tif'
bucket_directory = 'pred_batch_2/'

for blob in blobs:
    if blob.name.endswith('.tif'):
        try:
            print(f"Processing {blob.name}")
            blob.download_to_filename(tiff_filepath)

            # Open the .tif file and extract the metadata
            with rasterio.open(tiff_filepath) as src:
                original_meta = src.meta

            print(original_meta)
            # Perform prediction
            m = open_tiff(tiff_filepath, display_im=False)
            m = normalize_to_8bit(m)
            # Assuming m is normalized to [0, 1] range
            pred_map_full = full_prediction_tiff(m, None, model_list, seg_connector)

            # Convert predictions to binary (0 or 1) and then cast to int8
            pred_map_full = (pred_map_full > 0.5).astype('int8')

            try:
              mask = (m == 0)
              pred_map_full = pred_map_full * ~mask

            except:
              mask = (m[:,:,0] == 0)
              pred_map_full = pred_map_full * ~mask

            # Add a new dimension to represent single band if needed
            if pred_map_full.ndim == 2:
                pred_map_full = np.expand_dims(pred_map_full, axis=0)

            # Update metadata for new file
            new_meta = original_meta.copy()
            new_meta['dtype'] = 'int8'
            new_meta['count'] = pred_map_full.shape[0]
            new_meta['compress'] = 'lzw'

            # Create new file name for prediction
            new_file_name = local_path + blob.name.replace('.tif', '-pred-v1.tif')

            # Ensure the directory exists before attempting to write the file
            os.makedirs(os.path.dirname(new_file_name), exist_ok=True)

            # Write new file with updated metadata and prediction data
            with rasterio.open(new_file_name, 'w', **new_meta) as dest:
                dest.write(pred_map_full)

            # Upload the prediction back to the bucket
            pred_blob = bucket.blob(bucket_directory + blob.name.replace('.tif', '-pred-v1.tif'))
            pred_blob.upload_from_filename(new_file_name)

        except Exception as e:
            print(f"Prediction failed for file: {blob.name}. Error: {str(e)}")
            traceback.print_exc()

        finally:
            # Delete the local files to free up memory
            if os.path.isfile(tiff_filepath):
                os.remove(tiff_filepath)
            if os.path.isfile(new_file_name):
                os.remove(new_file_name)

        print(f"Processing of {blob.name} complete.")


# In[ ]:


try:
  mask = (m == 0)
  pred_map_full = pred_map_full * ~mask

except:
  mask = (m[:,:,0] == 0)
  pred_map_full = pred_map_full * ~mask


# In[ ]:


original_meta


# In[ ]:


pred_map_full = np.squeeze(pred_map_full)


# In[ ]:


display(pred_map_full[::5, ::5])


# ## Prediction Individual Tiffs

# In[ ]:


get_ipython().system('ls "/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/New_Data/for_mason/need_buffering"')


# In[ ]:


input_tif_fp = "/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/New_Data/for_mason/need_buffering/sn2_VIS.tif"
input = open_tiff(input_tif_fp,display_im=False)
input = normalize_to_8bit(input)
with rasterio.open(input_tif_fp) as src:
    original_meta = src.meta
    print(original_meta)
stats(input)


# In[ ]:


display(input[::20, ::20]) ## downscale


# In[ ]:


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


# In[ ]:


display(pred_map)
#download_tiff(pred_map,original_meta, filename='sn2_pred.tif')


# In[ ]:


display(pred_map[4000:7000, 4000:7000])


# In[ ]:


import os
def count_files_in_directory(directory_path):
    with os.scandir(directory_path) as entries:
        return sum(1 for entry in entries if entry.is_file())

directory_path = "/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/New_Data/seg_connector_tiles/PredV3/mask"
file_count = count_files_in_directory(directory_path)
print(f"Number of files in directory: {file_count}")


# In[ ]:


gt_tif_fp = "/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/New_Data/for_mason/need_buffering/sn2_gt.tif"
gt = open_tiff(input_tif_fp,display_im=False)
gt = normalize_to_8bit(gt)


# In[ ]:


display(gt)

