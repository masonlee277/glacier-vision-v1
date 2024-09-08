import os
import pytest
from fastapi.testclient import TestClient
import asyncio
from PIL import Image
import io
import logging
import numpy as np

# Add the project root to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.app import app

# Set up logging
log_dir = os.path.join('data', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'test_api_logs.txt')

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

logger.info("Test API logging initialized. Previous logs cleared.")

@pytest.fixture(scope="module")
def test_client():
    logger.info("Creating test client")
    return TestClient(app)

@pytest.fixture(scope="module")
def test_tiff_file():
    tiff_path = os.path.join(
        'data', 'mark_validation',
        'clip_WV03_20220801143842_1040010079411F00_22AUG01143842-M1BS-506796344080_01_P001_u16rf3413_RGB_COMP_CROPPED.tif'
    )
    assert os.path.exists(tiff_path), f"TIFF file not found at {tiff_path}"
    return tiff_path

@pytest.fixture(scope="module")
def test_png_file():
    png_path = os.path.join('data', 'test_data', 'output.png')
    assert os.path.exists(png_path), f"PNG file not found at {png_path}"
    return png_path

def test_upload_endpoint(test_client, test_tiff_file):
    logger.info("Starting test_upload_endpoint")
    
    with open(test_tiff_file, "rb") as tiff_file:
        files = {"files": ("test_image.tif", tiff_file, "image/tiff")}
        
        response = test_client.post("/upload/", files=files)

    logger.info(f"Received response with status code: {response.status_code}")
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    
    response_data = response.json()
    assert "file_ids" in response_data, "Expected 'file_ids' in response"
    assert len(response_data["file_ids"]) > 0, "Expected at least one file ID in response"
    
    logger.info("test_upload_endpoint completed successfully")
    return response_data["file_ids"][0]

def test_predict_endpoint(test_client, test_tiff_file):
    logger.info("Starting test_predict_endpoint")
    
    with open(test_tiff_file, "rb") as tiff_file:
        files = {"file": ("test_image.tif", tiff_file, "image/tiff")}
        
        response = test_client.post("/predict/", files=files)

    logger.info(f"Received response with status code: {response.status_code}")
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    
    logger.info(f"Response content type: {response.headers['content-type']}")
    assert response.headers["content-type"] == "image/png", "Expected content-type to be image/png"

    # Verify the response is a valid PNG image
    try:
        logger.info("Attempting to open response content as an image")
        img = Image.open(io.BytesIO(response.content))
        logger.info(f"Successfully opened image: size={img.size}, mode={img.mode}")
    except IOError as e:
        logger.error(f"Failed to open response content as an image: {str(e)}")
        pytest.fail("Response content is not a valid image")

    logger.info("test_predict_endpoint completed successfully")

def test_predict_multiple_endpoint(test_client, test_upload_endpoint):
    logger.info("Starting test_predict_multiple_endpoint")
    
    file_id = test_upload_endpoint
    
    data = {"file_ids": [file_id]}
    
    response = test_client.post("/predict_multiple/", json=data)

    logger.info(f"Received response with status code: {response.status_code}")
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    
    response_data = response.json()
    assert "predictions" in response_data, "Expected 'predictions' in response"
    assert len(response_data["predictions"]) > 0, "Expected at least one prediction in response"
    
    logger.info("test_predict_multiple_endpoint completed successfully")
    return response_data["predictions"][0]["prediction_id"]

def test_get_prediction_endpoint(test_client, test_predict_multiple_endpoint):
    logger.info("Starting test_get_prediction_endpoint")
    
    prediction_id = test_predict_multiple_endpoint
    
    response = test_client.get(f"/prediction/{prediction_id}")

    logger.info(f"Received response with status code: {response.status_code}")
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    
    logger.info(f"Response content type: {response.headers['content-type']}")
    assert response.headers["content-type"] == "image/png", "Expected content-type to be image/png"

    # Verify the response is a valid PNG image
    try:
        logger.info("Attempting to open response content as an image")
        img = Image.open(io.BytesIO(response.content))
        logger.info(f"Successfully opened image: size={img.size}, mode={img.mode}")
    except IOError as e:
        logger.error(f"Failed to open response content as an image: {str(e)}")
        pytest.fail("Response content is not a valid image")

    logger.info("test_get_prediction_endpoint completed successfully")

def test_connect_rivers_endpoint(test_client, test_png_file):
    logger.info("Starting test_connect_rivers_endpoint")
    
    with open(test_png_file, "rb") as png_file:
        files = {"file": ("test_image.png", png_file, "image/png")}
        
        response = test_client.post("/connect_rivers/", files=files)

    logger.info(f"Received response with status code: {response.status_code}")
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    
    logger.info(f"Response content type: {response.headers['content-type']}")
    assert response.headers["content-type"] == "image/png", "Expected content-type to be image/png"

    # Verify the response is a valid PNG image
    try:
        logger.info("Attempting to open response content as an image")
        img = Image.open(io.BytesIO(response.content))
        logger.info(f"Successfully opened image: size={img.size}, mode={img.mode}")
        
        # Additional checks for the connected rivers image
        assert img.mode == "L", f"Expected grayscale image, but got mode {img.mode}"
        
        # Check if the image contains any white pixels (connected rivers)
        img_array = np.array(img)
        assert np.any(img_array > 0), "No connected rivers found in the image"
        
        logger.info("Connected rivers image verified successfully")
    except IOError as e:
        logger.error(f"Failed to open response content as an image: {str(e)}")
        pytest.fail("Response content is not a valid image")
    except AssertionError as e:
        logger.error(f"Image verification failed: {str(e)}")
        pytest.fail(str(e))

    logger.info("test_connect_rivers_endpoint completed successfully")

if __name__ == "__main__":
    pytest.main([__file__])