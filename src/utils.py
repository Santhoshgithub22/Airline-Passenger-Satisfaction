from src.logger import logging
from src.exception import CustomException
import os, sys
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_objt:
            return pickle.load(file_objt)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            ## Train Model
            model.fit(X_train, y_train)

            ## Predict Testing Data
            y_test_pred = model.predict(X_test)

            ## Get Accuracy, F1 Score, Precision & Recall for train and test data
            # train_model_score = r2_score(y_test, y_train_pred)

            model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
            model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score
            model_test_precision = precision_score(y_test, y_test_pred) # Calculate Precision
            model_test_recall = recall_score(y_test, y_test_pred) # Calculate Recall

            report[list(models.keys())[i]] = model_test_accuracy

        return report
    
    except Exception as e:
        logging.info("Exception occured during model training")
        raise CustomException(e, sys)
    