from youtubeComments.logger import logging
from youtubeComments.exception import CustomException
from youtubeComments.components.data_ingestion import DataIngestion
from youtubeComments.components.data_transforamation import DataTransformationArtifacts,DataTransformation
from youtubeComments.entity.config_entity import DataIngestionConfig,DataTransformationConfig
from youtubeComments.entity.artifact_entity import DataIngestionArtifacts,DataTransformationArtifacts

import sys
class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
    


    def start_data_ingestion(self):
        try:
            logging.info("Data ingestion started.")
            data_ingestion = DataIngestion()
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed.")
            return data_ingestion_artifacts
        except CustomException as e:
            logging.error(f"Error occurred during data ingestion: {str(e)}")
    def start_data_transformation(self, data_ingestion_artifacts = DataIngestionArtifacts) -> DataTransformationArtifacts:
        logging.info("Entered the start_data_transformation method of TrainPipeline class")
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifacts = data_ingestion_artifacts,
                data_transformation_config=self.data_transformation_config
            )

            data_transformation_artifacts = data_transformation.initiate_data_transformation()
            
            logging.info("Exited the start_data_transformation method of TrainPipeline class")
            return data_transformation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
        

    def run_pipeline(self):
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            logging.info("Data ingestion artifacts: {0}".format(data_ingestion_artifacts))
            data_transformation_artifacts = self.start_data_transformation(
                data_ingestion_artifacts=data_ingestion_artifacts
            )
        except Exception as e:
            logging.error(f"Error occurred during pipeline execution: {str(e)}")
            print(e)
