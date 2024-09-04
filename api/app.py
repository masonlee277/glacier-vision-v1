import os
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import logging
from datetime import datetime

# Import your utility functions
# add utils to the path 
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)



from utils import normalize_to_8bit, full_prediction_tiff, compile_model, mean_iou, dice_lossV1

# Set up logging
log_dir = os.path.join('data', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'api_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()

# Helper function to load models
def load_models():
    logger.info("Starting to load models")
    
    # Load RiverNet models
    model_weights_dir = "data/model_weights/riverNet/RiverNet_checkpoint_dir/retrained"
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
    seg_connector_path = 'data/model_weights/segConnector/wandb_artifacts/model-training_on_RiverNet_PredictionsV2:v29'
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
    riverNet_models, seg_connector = load_models()
except Exception as e:
    logger.critical(f"Failed to load models at startup. Error: {str(e)}")
    raise

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    logger.info(f"Received prediction request for file: {file.filename}")
    
    try:
        # Read and preprocess the image
        contents = await file.read()
        logger.debug(f"Read file contents, size: {len(contents)} bytes")
        
        image = Image.open(io.BytesIO(contents))
        logger.debug(f"Opened image, format: {image.format}, size: {image.size}, mode: {image.mode}")
        
        image_array = np.array(image)
        logger.debug(f"Converted image to numpy array, shape: {image_array.shape}, dtype: {image_array.dtype}")
        
        # Normalize the image
        normalized_image = normalize_to_8bit(image_array)
        logger.debug(f"Normalized image, new shape: {normalized_image.shape}, dtype: {normalized_image.dtype}")
        
        # Make prediction
        logger.info("Starting prediction process")
        prediction = full_prediction_tiff(normalized_image, None, riverNet_models, seg_connector)
        logger.info("Prediction completed")
        logger.debug(f"Prediction shape: {prediction.shape}, dtype: {prediction.dtype}")
        
        # Convert prediction to binary
        binary_prediction = (prediction > 0.5).astype(np.uint8) * 255
        logger.debug(f"Converted prediction to binary, shape: {binary_prediction.shape}, dtype: {binary_prediction.dtype}")
        
        # Save the prediction as an image
        output_image = Image.fromarray(binary_prediction)
        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        logger.info("Saved prediction as PNG image")
        
        logger.info("Prediction process completed successfully")
        return FileResponse(output_buffer, media_type="image/png", filename="prediction.png")
    
    except Exception as e:
        logger.error(f"Error during prediction process: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

@app.on_event("startup")
async def startup_event():
    logger.info("API is starting up")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API is shutting down")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)