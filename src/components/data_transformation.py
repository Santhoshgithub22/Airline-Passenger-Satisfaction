import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformConfig:
    preprocess_obj_file_path = os.path.join("artifacts", "preprocessor.pkl") #It will be a folder and file name

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformConfig()

    def get_transformation_object(self):

        try:
            logging.info("Data Transformation Initiated")

            ## Separating Numerical features

            numerical_features = ['Age', 'Flight Distance', 'Inflight wifi service','Departure/Arrival time convenient',
                                            'Ease of Online booking','Gate location', 'Food and drink', 'Online boarding',
                                            'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 
                                            'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
                                            'Departure Delay in Minutes', 'Arrival Delay in Minutes']
                        
            ordinal_features=["Customer Type", "Class"]

            onehot_features = ["Gender", "Type of Travel"]

            # Define the transformers
            numeric_transformer = StandardScaler()
            ordinal_transformer = OrdinalEncoder()
            oh_transformer = OneHotEncoder(drop='first')


            ## We created numericals separately and categoricals separately,Now we need to combine this.
            preprocessor = ColumnTransformer(
                transformers=[
                    ('ordinal', ordinal_transformer, ordinal_features),
                    ('oh', oh_transformer, onehot_features),
                    ('num', numeric_transformer, numerical_features)
                ]
            )
                                    
            logging.info('pipeline completed')
            return preprocessor

        except Exception as e:
            logging.info("Error occured in Data Transformation")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info("Read train set and test set is started")

            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Replacing null values")
            train_data["Arrival Delay in Minutes"] = train_data["Arrival Delay in Minutes"].fillna(train_data["Arrival Delay in Minutes"].median())
            test_data["Arrival Delay in Minutes"] = test_data["Arrival Delay in Minutes"].fillna(test_data["Arrival Delay in Minutes"].median())
            logging.info("Succesfully Null values replaced")

            logging.info("Dropping Unnamed Column")
            train_data = train_data.drop(["Unnamed: 0"], axis=1)
            test_data = test_data.drop(["Unnamed: 0"], axis=1)
            logging.info("Succesfully Unnamed column dropped")

            logging.info("Mapping into numerical values for dependent columns")

            satisfaction_mapping = {'satisfied': 1,'neutral or dissatisfied': 0}

            train_data['satisfaction'] = train_data['satisfaction'].map(satisfaction_mapping)
            test_data['satisfaction'] = test_data['satisfaction'].map(satisfaction_mapping)
            
            logging.info("Read train set and test set is completed")

            logging.info(f'Train Dataframe Head: \n{train_data.head().to_string()}')
            logging.info(f"Test Dataframe Head: \n{test_data.head().to_string()}")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_transformation_object()

            logging.info("Splitting the dataset into training set and test set")
            

            target_column_name = "satisfaction"
            drop_columns = target_column_name

            input_feature_train_df = train_data.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_data[target_column_name]

            input_feature_test_df=test_data.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_data[target_column_name]

            ## Transforming using preprocessor object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] # We are just concatenating our input train and output train
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] # we are just concatenating our input test and output test
            # above 2 lines edhukku use panni irukom na, data transformation panna aprm neraya rows and columns vara chance irukku
            # data va input ah kudukurappa, again modhala irukku run aaga vida mudiyadhu, so just concatenating with array
            # idhu pandrapa it will run very quickly.

            save_object(

                file_path=self.data_transformation_config.preprocess_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_obj_file_path,
            )

        except Exception as e:
            logging.info("Error occured in the initiate data_transformation")
            raise CustomException(e, sys)