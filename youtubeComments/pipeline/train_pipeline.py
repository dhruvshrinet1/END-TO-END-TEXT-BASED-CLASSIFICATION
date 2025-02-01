from youtubeComments.logger import logging
from youtubeComments.exception import CustomException
from youtubeComments.components.data_ingestion import DataIngestion

from youtubeComments.entity.config_entity import DataIngestionConfig
from youtubeComments.entity.artifact_entity import DataIngestionArtifacts


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    


    def start_data_ingestion(self):
        try:
            logging.info("Data ingestion started.")
            data_ingestion = DataIngestion()
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed.")
            return data_ingestion_artifacts
        except CustomException as e:
            logging.error(f"Error occurred during data ingestion: {str(e)}")

    def run_pipeline(self):
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            logging.info("Data ingestion artifacts: {0}".format(data_ingestion_artifacts))
        except Exception as e:
            logging.error(f"Error occurred during pipeline execution: {str(e)}")
            print(e)
