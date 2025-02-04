import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import os
import pickle
import sys
# TensorFlow and Keras imports
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences


# NLTK imports
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from youtubeComments.exception import CustomException
from youtubeComments.logger import logging
from youtubeComments.entity.config_entity import DataTransformationConfig,ModelTrainerConfig
from youtubeComments.entity.artifact_entity import DataTransformationArtifacts,DataIngestionArtifacts,ModelTrainerArtifacts

from youtubeComments.constants import *

from youtubeComments.ml.model import ModelArchitecture



class ModelTrainer:
    def __init__(self,data_transformation_artifacts:DataIngestionArtifacts,model_trainer_config:ModelTrainerConfig):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def spliting_data(self,csv_path):
        try:
            df = pd.read_csv(csv_path)
            X = df['Comment']
            y = df['Sentiment']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.model_trainer_config.RANDOM_STATE)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in splitting data: {str(e)}")
            raise CustomException("Error in splitting data")       
         
    def tokenizing(self,x_train):
        try:
                # Convert all elements in x_train to strings and handle NaN
                x_train = x_train.astype(str)  # Convert non-string values to strings
                x_train = x_train.fillna('')  # Replace NaN with empty strings
                logging.info("Applying tokenization on the data")
                tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
                tokenizer.fit_on_texts(x_train)
                sequences = tokenizer.texts_to_sequences(x_train)
                logging.info(f"converting text to sequences: {sequences}")
                sequences_matrix = pad_sequences(sequences,maxlen=self.model_trainer_config.MAX_LEN)
                logging.info(f" The sequence matrix is: {sequences_matrix}")
                return sequences_matrix,tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e


    def initiate_model_trainer(self,) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            logging.info("Entered the initiate_model_trainer function ")
            x_train,x_test,y_train,y_test = self.spliting_data(csv_path='artifacts/02_03_2025_14_23_10/DataTransformationArtifacts/final.csv')
            model_architecture = ModelArchitecture()   

            model = model_architecture.get_model()



            logging.info(f"Xtrain size is : {x_train.shape}")

            logging.info(f"Xtest size is : {x_test.shape}")

            sequences_matrix,tokenizer =self.tokenizing(x_train)
            from sklearn.preprocessing import LabelEncoder

# Initialize encoder
            encoder = LabelEncoder()

            # Fit and transform labels
            y_train = encoder.fit_transform(y_train)

            # Convert to int32 (required by Keras)
            y_train = y_train.astype('int32')

            print(y_train.dtype)  # Should now print 'int32'


            logging.info("Entered into model training")
            print(sequences_matrix.dtype)  # Should be float32 or int32, NOT 'object'
            print(y_train.dtype)  # Should be int32 or float32, NOT 'object'
            model.fit(sequences_matrix, y_train, 
                        batch_size=self.model_trainer_config.BATCH_SIZE, 
                        epochs = self.model_trainer_config.EPOCH, 
                        validation_split=self.model_trainer_config.VALIDATION_SPLIT, 
                        )
            logging.info("Model training finished")

            
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR,exist_ok=True)



            logging.info("saving the model")
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)

            x_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH)

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path = self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path = self.model_trainer_config.Y_TEST_DATA_PATH)
            logging.info("Returning the ModelTrainerArtifacts")
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e



        