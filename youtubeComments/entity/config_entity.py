from dataclasses import dataclass
from youtubeComments.constants import *
import os

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME = BUCKET_NAME
        self.ZIP_FILE_NAME = ZIP_FILE_NAME
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,self.ZIP_FILE_NAME)


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR:str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.LABEL = LABEL
        self.TRAIN_DATA_PATH = os.path.join(os.getcwd(),ARTIFACTS_DIR,"train.csv")
        self.TEST_DATA_PATH = os.path.join(os.getcwd(),ARTIFACTS_DIR,"test.csv")
        self.TRANSFORMED_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,TRANSFORMED_FILE_NAME)

        self.TRANSFORMED_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,TRANSFORMED_FILE_NAME)


@dataclass
class ModelTrainerConfig: 
    def __init__(self):
        self.TRAINED_MODEL_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR) 
        self.TRAINED_MODEL_PATH = os.path.join(self.TRAINED_MODEL_DIR,TRAINED_MODEL_NAME)
        self.X_TEST_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, X_TEST_FILE_NAME)
        self.Y_TEST_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, Y_TEST_FILE_NAME)
        self.X_TRAIN_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, X_TRAIN_FILE_NAME)
        self.MAX_WORDS = MAX_WORDS
        self.MAX_LEN = MAX_LEN
        self.LOSS = LOSS
        self.METRICS = METRICS
        self.ACTIVATION = ACTIVATION
        self.LABEL = LABEL
        self.RANDOM_STATE = RANDOM_STATE
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.VALIDATION_SPLIT = VALIDATION_SPLIT


