import os
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io

# Import your utility functions
from utils import normalize_to_8bit, full_prediction_tiff, compile_model, mean_iou, dice_lossV1

app = FastAPI()

# Helper function to load models
def load_models():
    # Load RiverNet models
    model_weights_dir = "data/model_weights/riverNet/RiverNet_checkpoint_dir/retrained"
    checkpoints = [
        os.path.join(model_weights_dir, f"model_weights_epoch_{epoch}.h5")
        for epoch in [80, 70, 90, 100]
    ]
    
    riverNet_models = []
    for checkpoint in checkpoints:
        model = compile_model(512, 512)
        model.load_weights(checkpoint)
        riverNet_models.append(model)
    
    # Load SegConnector model
    seg_connector = tf.keras.models.load_model(
        'data/model_weights/segConnector/wandb_artifacts/model-training_on_RiverNet_PredictionsV2:v29',
        custom_objects={'mean_iou': mean_iou, 'dice_loss': dice_lossV1}
    )
    
    return riverNet_models, seg_connector

# Load models at startup
riverNet_models, seg_connector = load_models()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_array = np.array(image)
    
    # Normalize the image
    normalized_image = normalize_to_8bit(image_array)
    
    # Make prediction
    prediction = full_prediction_tiff(normalized_image, None, riverNet_models, seg_connector)
    
    # Convert prediction to binary
    binary_prediction = (prediction > 0.5).astype(np.uint8) * 255
    
    # Save the prediction as an image
    output_image = Image.fromarray(binary_prediction)
    output_buffer = io.BytesIO()
    output_image.save(output_buffer, format="PNG")
    output_buffer.seek(0)
    
    return FileResponse(output_buffer, media_type="image/png", filename="prediction.png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)