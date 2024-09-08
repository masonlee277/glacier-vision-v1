import os
import sys
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
from typing import List
from pydantic import BaseModel
from utils.evaluation_utils import process_and_predict_tiff

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import utility functions
from utils import full_prediction_tiff, compile_model, mean_iou, dice_lossV1
from utils.logger_utils import Logger

# Initialize logger
logger = Logger('api')

app = FastAPI()

# In-memory storage for uploaded files and predictions
file_storage = {}
prediction_storage = {}

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

@contextmanager
def temporary_file(suffix='.tif'):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        yield temp_file
    finally:
        temp_file.close()
        os.unlink(temp_file.name)

class PredictionRequest(BaseModel):
    file_ids: List[str]

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    logger.info(f"Received upload request for {len(files)} files")
    
    uploaded_file_ids = []
    for file in files:
        file_id = str(uuid.uuid4())
        contents = await file.read()
        file_storage[file_id] = contents
        uploaded_file_ids.append(file_id)
        logger.info(f"File uploaded: {file.filename}, ID: {file_id}")
    
    return {"message": "Files uploaded successfully", "file_ids": uploaded_file_ids}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    logger.info(f"Received prediction request for file: {file.filename}")
    
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name
        
        logger.info(f"Saved uploaded file to temporary location: {temp_file_path}")
        
        # Process and predict
        output_path, unique_filename = process_and_predict_tiff(temp_file_path, riverNet_models, seg_connector)
        
        logger.info("Prediction process completed successfully")
        return FileResponse(output_path, media_type="image/png", filename=unique_filename)
    
    except Exception as e:
        logger.error(f"Error during prediction process: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {str(e)}")
    finally:
        # Clean up the temporary file
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
            logger.info(f"Cleaned up temporary file: {temp_file_path}")

@app.post("/predict_multiple/")
async def predict_multiple(request: PredictionRequest):
    logger.info(f"Received prediction request for multiple files")
    
    predictions = []
    for file_id in request.file_ids:
        if file_id not in file_storage:
            raise HTTPException(status_code=400, detail=f"File ID {file_id} not found")
        
        contents = file_storage[file_id]
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
                temp_file.write(contents)
                temp_file_path = temp_file.name
            
            logger.info(f"Processing file ID: {file_id}")
            
            output_path, unique_filename = process_and_predict_tiff(temp_file_path, riverNet_models, seg_connector)
            
            prediction_id = os.path.splitext(unique_filename)[0]
            prediction_storage[prediction_id] = output_path
            predictions.append({"file_id": file_id, "prediction_id": prediction_id})
            
        except Exception as e:
            logger.error(f"Error during prediction process for file {file_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {str(e)}")
        finally:
            # Clean up the temporary file
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
    
    logger.info("Prediction process completed successfully for all files")
    return {"message": "Predictions completed", "predictions": predictions}

@app.get("/prediction/{prediction_id}")
async def get_prediction(prediction_id: str):
    logger.info(f"Received request for prediction ID: {prediction_id}")
    
    if prediction_id not in prediction_storage:
        raise HTTPException(status_code=404, detail=f"Prediction ID {prediction_id} not found")
    
    output_path = prediction_storage[prediction_id]
    logger.info(f"Retrieving prediction from: {output_path}")
    
    try:
        return FileResponse(output_path, media_type="image/png", filename=os.path.basename(output_path))
    except Exception as e:
        logger.error(f"Error retrieving prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error retrieving prediction: {str(e)}")
    finally:
        # Remove the prediction from storage
        del prediction_storage[prediction_id]
        logger.info(f"Removed prediction ID {prediction_id} from storage")

@app.on_event("startup")
async def startup_event():
    logger.info("API is starting up")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API is shutting down")

if __name__ == "__main__":
    logger.info("Starting the API server")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")