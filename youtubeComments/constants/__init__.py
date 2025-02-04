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


MODEL_TRAINER_ARTIFACTS_DIR = 'ModelTrainerArtifacts'
TRAINED_MODEL_DIR = 'trained_model'
TRAINED_MODEL_NAME = 'model.h5'
X_TEST_FILE_NAME = 'x_test.csv'
Y_TEST_FILE_NAME = 'y_test.csv'

X_TRAIN_FILE_NAME = 'x_train.csv'

RANDOM_STATE = 42
EPOCH = 10
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2

# Model Architecture constants
MAX_WORDS = 10000
MAX_LEN = 100
LOSS = 'binary_crossentropy'
METRICS = ['accuracy']
ACTIVATION = 'sigmoid'

