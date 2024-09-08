import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    MODEL_WEIGHTS_DIR: str = os.path.join(PROJECT_ROOT, "data/model_weights/riverNet/retrained")
    SEG_CONNECTOR_PATH: str = os.path.join(PROJECT_ROOT, 'data/model_weights/segConnector/wandb_artifacts/model-training_on_own_predictions_v35')
    OUTPUT_DIR: str = os.path.join(PROJECT_ROOT, 'data', 'outputs', 'pred')
    UPLOAD_DIR: str = os.path.join(PROJECT_ROOT, 'data', 'uploads')
    LOG_DIR: str = os.path.join(PROJECT_ROOT, 'data', 'logs')

    class Config:
        env_file = ".env"

settings = Settings()