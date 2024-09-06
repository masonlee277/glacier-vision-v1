import os
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
import asyncio
from PIL import Image
import io
import logging
import shutil

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

@pytest.mark.asyncio
async def test_predict_endpoint(test_client):
    logger.info("Starting test_predict_endpoint")
    
    tiff_path = os.path.join(
        'data', 'mark_validation',
        'clip_WV03_20220801143842_1040010079411F00_22AUG01143842-M1BS-506796344080_01_P001_u16rf3413_RGB_COMP_CROPPED.tif'
    )
    
    logger.info(f"TIFF file path: {tiff_path}")
    assert os.path.exists(tiff_path), f"TIFF file not found at {tiff_path}"
    logger.info("TIFF file exists")

    with open(tiff_path, "rb") as tiff_file:
        logger.info("Opened TIFF file")
        file_size = os.path.getsize(tiff_path)
        logger.info(f"TIFF file size: {file_size} bytes")
        
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

    # Save the response content to a specific file, overwriting if it exists
    output_dir = os.path.join('data', 'outputs', 'test')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'test_file.png')
    
    with open(output_file, 'wb') as f:
        f.write(response.content)
    logger.info(f"Prediction result saved to: {output_file}")
    print(f"Prediction result saved to: {output_file}")

    # Validate that we can get output
    assert os.path.exists(output_file), f"Output file not found at {output_file}"
    assert os.path.getsize(output_file) > 0, "Output file is empty"

    # Verify that the saved file is a valid image
    try:
        with Image.open(output_file) as img:
            logger.info(f"Successfully opened saved image: size={img.size}, mode={img.mode}")
    except IOError as e:
        logger.error(f"Failed to open saved image: {str(e)}")
        pytest.fail("Saved file is not a valid image")

    logger.info("test_predict_endpoint completed successfully")

if __name__ == "__main__":
    logger.info("Running test_api.py")
    pytest.main([__file__])
    logger.info("Finished running test_api.py")