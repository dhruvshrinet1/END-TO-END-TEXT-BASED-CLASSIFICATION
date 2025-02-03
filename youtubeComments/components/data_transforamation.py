import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import os
# TensorFlow and Keras imports
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# NLTK imports
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from youtubeComments.exception import CustomException
from youtubeComments.logger import logging
from youtubeComments.entity.config_entity import DataTransformationConfig
from youtubeComments.entity.artifact_entity import DataTransformationArtifacts,DataIngestionArtifacts



class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,data_ingestion_artifacts:DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    def clean_data(self,text):
        text = str(text).lower()  # Convert to lowercase
        text = re.sub(r'http\S+', '', text)  # Remove links
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#\w+', '', text)  # Remove hashtags
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        return text.strip()

    def preprocess_text(self, df):
        stop_words = set(stopwords.words('english'))
        df['Comment'] = df['Comment'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
        return df


    def initiate_data_transformation(self):
        try:
            logging.info("Data transformation started.")

            # Load the dataset
            df = pd.read_csv('notebook/data/YoutubeCommentsDataSet.csv')




            # Preprocess the text data
            df['Comment'] = df['Comment'].apply(self.clean_data)
            df = self.preprocess_text(df)
            logging.info("Data preprocessing completed.")
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR,exist_ok=True)
            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH, index=False, header=True)

            data_transformation_artifact = DataTransformationArtifacts(
                transformed_data_path=self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR
            )
            logging.info("Data transformation completed successfully.")
            return data_transformation_artifact 
        except CustomException as e:
            logging.error(f"Error occurred during data transformation: {str(e)}")
            

