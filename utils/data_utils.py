# data_utils.py
# data_utils.py
from .common_imports import *
from .common_utils import *
from .model_utils import *
from .evaluation_utils import *
from .training_utils import *
from .image_utils import *
from .visualization_utils import *



import numpy as np
import rasterio
from rasterio.transform import from_origin

def download_tiff(array, original_meta, filepath='/content/', filename='output.tif', transform=None, crs=None):
    """
    Saves a NumPy array as a compressed TIFF file.

    Parameters:
    array (np.array): The NumPy array to be saved.
    filepath (str): The directory path where the TIFF file will be saved. Defaults to '/content/'.
    filename (str): The name of the TIFF file. Defaults to 'output.tif'.
    transform (rasterio.transform.Affine): Optional. Affine transform for the TIFF file. Defaults to None.
    crs (rasterio.crs.CRS): Optional. Coordinate Reference System for the TIFF file. Defaults to None.
    """
    # Ensure the directory path ends with a slash
    if not filepath.endswith('/'):
        filepath += '/'

    # Create full path
    full_path = filepath + filename

    # If transform is not provided, use a default one
    if transform is None:
        transform = from_origin(0, 0, 1, 1)

    # If crs is not provided, use a default one
    if crs is None:
        crs = rasterio.crs.CRS.from_epsg(4326)  # WGS84

    # Add a new dimension to represent single band if needed
    if len(array.shape) == 2:
        array = np.expand_dims(array, axis=0)

    new_meta = original_meta.copy()
    new_meta['dtype'] = 'int8'
    new_meta['count'] = array.shape[0]
    new_meta['compress'] = 'lzw'


    # Write the TIFF file
    with rasterio.open(full_path, 'w', **new_meta) as dst:
        dst.write(array)

# Example usage
# array = np.random.rand(100, 100)  # Example array
