import os
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
import asyncio
from PIL import Image
import io

# Add the project root to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.app import app

@pytest.fixture(scope="module")
def test_client():
    return TestClient(app)

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.mark.asyncio
async def test_predict_endpoint(test_client):
    tiff_path = os.path.join(
        'data', 'mark_validation',
        'clip_WV03_20220801143842_1040010079411F00_22AUG01143842-M1BS-506796344080_01_P001_u16rf3413_RGB_COMP_CROPPED.tif'
    )
    
    assert os.path.exists(tiff_path), f"TIFF file not found at {tiff_path}"

    with open(tiff_path, "rb") as tiff_file:
        files = {"file": ("test_image.tif", tiff_file, "image/tiff")}
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/predict/", files=files)

    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    assert response.headers["content-type"] == "image/png", "Expected content-type to be image/png"

    # Verify the response is a valid PNG image
    try:
        Image.open(io.BytesIO(response.content))
    except IOError:
        pytest.fail("Response content is not a valid image")

    # Additional assertions can be added here to verify the prediction results

if __name__ == "__main__":
    pytest.main([__file__])