import os
import sys
import logging
from datetime import datetime
import io
import tempfile
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import rasterio
from utils.image_utils import open_tiff, normalize_to_8bit
from contextlib import contextmanager
import uuid

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import utility functions
from utils import full_prediction_tiff, compile_model, mean_iou, dice_lossV1

# Set up logging
log_dir = os.path.join('data', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'app_logs.log')

# Clear the log file if it exists
if os.path.exists(log_file):
    open(log_file, 'w').close()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

logger.info("Logging initialized. Previous logs cleared.")

app = FastAPI()

# Helper function to load models
def load_models():
    logger.info("Starting to load models")
    
    # Load RiverNet models
    model_weights_dir = os.path.join(project_root, "data/model_weights/riverNet/retrained")
    checkpoints = [
        os.path.join(model_weights_dir, f"model_weights_epoch_{epoch}.h5")
        for epoch in [80, 70, 90, 100]
    ]
    
    riverNet_models = []
    for checkpoint in checkpoints:
        logger.debug(f"Loading RiverNet model from checkpoint: {checkpoint}")
        try:
            model = compile_model(512, 512)
            model.load_weights(checkpoint)
            riverNet_models.append(model)
            logger.info(f"Successfully loaded RiverNet model from {checkpoint}")
        except Exception as e:
            logger.error(f"Failed to load RiverNet model from {checkpoint}. Error: {str(e)}")
            raise
    
    # Load SegConnector model
    seg_connector_path = os.path.join(project_root, 'data/model_weights/segConnector/wandb_artifacts/model-training_on_own_predictions_v35')
    logger.debug(f"Loading SegConnector model from: {seg_connector_path}")
    try:
        seg_connector = tf.keras.models.load_model(
            seg_connector_path,
            custom_objects={'mean_iou': mean_iou, 'dice_loss': dice_lossV1}
        )
        logger.info("Successfully loaded SegConnector model")
    except Exception as e:
        logger.error(f"Failed to load SegConnector model. Error: {str(e)}")
        raise
    
    logger.info("All models loaded successfully")
    return riverNet_models, seg_connector

# Load models at startup
try:
    logger.info("Attempting to load models at startup")
    riverNet_models, seg_connector = load_models()
    logger.info("Models successfully loaded at startup")
except Exception as e:
    logger.critical(f"Failed to load models at startup. Error: {str(e)}")
    raise

# Reduce multipart debug logging
logging.getLogger("multipart").setLevel(logging.WARNING)

@contextmanager
def temporary_file(suffix='.tif'):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        yield temp_file
    finally:
        temp_file.close()
        os.unlink(temp_file.name)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    logger.info(f"Received prediction request for file: {file.filename}")
    
    try:
        # Read the file contents
        contents = await file.read()
        logger.info(f"Read file contents, size: {len(contents)} bytes")
        
        # Save the contents to a temporary file
        with temporary_file() as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name
            logger.info(f"Saved contents to temporary file: {temp_file_path}")

            # Use open_tiff to read the image
            image_array = open_tiff(temp_file_path)
            if image_array is None:
                raise ValueError("Failed to open TIFF image")
            logger.info(f"Opened TIFF image, shape: {image_array.shape}, dtype: {image_array.dtype}")

            # Normalize the image
            normalized_image = normalize_to_8bit(image_array)
            logger.info(f"Normalized image, new shape: {normalized_image.shape}, dtype: {normalized_image.dtype}")
            
            # Make prediction
            logger.info("Starting prediction process")
            prediction = full_prediction_tiff(normalized_image, None, riverNet_models, seg_connector)
            logger.info(f"Prediction completed, shape: {prediction.shape}, dtype: {prediction.dtype}")
            
            # Convert prediction to binary
            binary_prediction = (prediction > 0.5).astype(np.uint8) * 255
            logger.info(f"Converted prediction to binary, shape: {binary_prediction.shape}, dtype: {binary_prediction.dtype}")
            
            # Save the prediction as an image
            output_dir = os.path.join('data', 'outputs', 'pred')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate a unique filename
            unique_filename = f"prediction_{uuid.uuid4().hex}.png"
            output_path = os.path.join(output_dir, unique_filename)
            
            output_image = Image.fromarray(binary_prediction.squeeze(), mode='L')
            output_image.save(output_path)
            logger.info(f"Saved prediction as PNG image: {output_path}")

        logger.info("Prediction process completed successfully")
        return FileResponse(output_path, media_type="image/png", filename=unique_filename)
    
    except Exception as e:
        logger.error(f"Error during prediction process: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {str(e)}")

@app.on_event("startup")
async def startup_event():
    logger.info("API is starting up")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API is shutting down")

if __name__ == "__main__":
    logger.info("Starting the API server")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")