import os
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
import asyncio
from PIL import Image
import io
import logging
import json

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
def event_loop():
    logger.info("Setting up event loop")
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    logger.info("Closing event loop")
    loop.close()

@pytest.fixture(scope="module")
def test_tiff_file():
    tiff_path = os.path.join(
        'data', 'mark_validation',
        'clip_WV03_20220801143842_1040010079411F00_22AUG01143842-M1BS-506796344080_01_P001_u16rf3413_RGB_COMP_CROPPED.tif'
    )
    assert os.path.exists(tiff_path), f"TIFF file not found at {tiff_path}"
    return tiff_path

@pytest.mark.asyncio
async def test_upload_endpoint(test_client, test_tiff_file):
    logger.info("Starting test_upload_endpoint")
    
    with open(test_tiff_file, "rb") as tiff_file:
        files = {"files": ("test_image.tif", tiff_file, "image/tiff")}
        
        logger.info("Sending POST request to /upload/ endpoint")
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/upload/", files=files)

    logger.info(f"Received response with status code: {response.status_code}")
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    
    response_data = response.json()
    assert "file_ids" in response_data, "Expected 'file_ids' in response"
    assert len(response_data["file_ids"]) > 0, "Expected at least one file ID in response"
    
    logger.info("test_upload_endpoint completed successfully")
    return response_data["file_ids"][0]

@pytest.mark.asyncio
async def test_predict_endpoint(test_client, test_tiff_file):
    logger.info("Starting test_predict_endpoint")
    
    with open(test_tiff_file, "rb") as tiff_file:
        files = {"file": ("test_image.tif", tiff_file, "image/tiff")}
        
        logger.info("Sending POST request to /predict/ endpoint")
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/predict/", files=files)

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

@pytest.mark.asyncio
async def test_predict_multiple_endpoint(test_client, test_upload_endpoint):
    logger.info("Starting test_predict_multiple_endpoint")
    
    file_id = await test_upload_endpoint
    
    data = {"file_ids": [file_id]}
    
    logger.info("Sending POST request to /predict_multiple/ endpoint")
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/predict_multiple/", json=data)

    logger.info(f"Received response with status code: {response.status_code}")
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    
    response_data = response.json()
    assert "predictions" in response_data, "Expected 'predictions' in response"
    assert len(response_data["predictions"]) > 0, "Expected at least one prediction in response"
    
    logger.info("test_predict_multiple_endpoint completed successfully")
    return response_data["predictions"][0]["prediction_id"]

@pytest.mark.asyncio
async def test_get_prediction_endpoint(test_client, test_predict_multiple_endpoint):
    logger.info("Starting test_get_prediction_endpoint")
    
    prediction_id = await test_predict_multiple_endpoint
    
    logger.info(f"Sending GET request to /prediction/{prediction_id} endpoint")
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get(f"/prediction/{prediction_id}")

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

if __name__ == "__main__":
    logger.info("Running test_api.py")
    pytest.main([__file__])
    logger.info("Finished running test_api.py")