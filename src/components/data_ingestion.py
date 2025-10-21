import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


#  Configuration dataclass for data ingestion
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


#  Main DataIngestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            #  Read the dataset
            df = pd.read_csv(os.path.join('notebook', 'data', 'stud.csv'))
            logging.info('Read the dataset as a dataframe')

            #  Create the artifacts directory if it doesn’t exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            #  Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train-test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    try:
        
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)
        logging.info(f" Data transformation completed successfully. Preprocessor saved at: {preprocessor_path}")

        model_trainer = ModelTrainer()
        r2_score_value = model_trainer.initiate_model_trainer(train_arr, test_arr)

        # Log final summary
        logging.info("\n" + "=" * 50)
        logging.info(" MODEL TRAINING COMPLETED SUCCESSFULLY!")
        logging.info(f" Final R2 Score: {r2_score_value:.4f}")
        logging.info(f" Model saved at: artifacts/model.pkl")
        logging.info("=" * 50 + "\n")

    except Exception as e:
        logging.error(" Error occurred during pipeline execution")
        logging.exception(e)
