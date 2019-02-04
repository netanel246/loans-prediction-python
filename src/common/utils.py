import pandas as pd
from server import abort
from sklearn.impute import SimpleImputer
from xgboost import DMatrix
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def impute_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    data: pd.DataFrame = df.copy()
    cols_with_missing = [col for col in data.columns if data[col].isnull().any()]
    for col in cols_with_missing:
        data[col + "_was_missing"] = data[col].isnull()

    imputer = SimpleImputer()
    imputed_train = pd.DataFrame(imputer.fit_transform(data))
    imputed_train.columns = data.columns
    return imputed_train


def parse_json(raw_json: str) -> DMatrix:
    if raw_json is None or raw_json == "":
        abort(500, message="raw_json is None or empty")
    row_to_predict = pd.read_json(path_or_buf=raw_json, orient="values", typ="series")
    row_to_predict_reshaped = row_to_predict.values.reshape(1, -1)
    return row_to_predict_reshaped


def balance_label_data(train_data: pd.DataFrame) -> pd.DataFrame:
    charged_off_count = len(train_data[train_data["Loan_Status"] == "Charged_Off"])
    charged_off_indices = train_data[train_data["Loan_Status"] == "Charged_Off"].index
    random_indices = np.random.choice(charged_off_indices, charged_off_count, replace=False)
    fully_paid_indices = train_data[train_data["Loan_Status"] == "Fully_Paid"].index
    under_sample_indices = np.concatenate([fully_paid_indices, random_indices])
    under_sample = train_data.loc[under_sample_indices]
    return under_sample


def report_metrics(y_test: pd.Series, predictions: np.ndarray):

    print(f"precision_score: {precision_score(y_test, predictions)}")
    print(f"recall_score: {recall_score(y_test, predictions)}")
    print(f"f1_score: {f1_score(y_test, predictions)}")
    print(f"confusion_matrix: {confusion_matrix(y_true=y_test, y_pred=predictions)}")
