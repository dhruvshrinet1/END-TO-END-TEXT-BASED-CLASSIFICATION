import os
from datetime import datetime


TIMESTAMP:str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts",TIMESTAMP)
BUCKET_NAME = "YOUTUBE"
ZIP_FILE_NAME = "archive(3).zip"
LABEL = "LABEL"


DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTransformationArtifacts'
TRANSFORMED_FILE_NAME = "final.csv"
