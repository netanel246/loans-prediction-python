import json
import pickle
from xgboost import DMatrix
import pandas as pd
from common import consts


class Model(object):
    __instance = None

    def __new__(cls):
        if Model.__instance is None:
            Model.__instance = object.__new__(cls)
        Model.__instance.model = cls.__get_model()
        return Model.__instance

    @staticmethod
    def __get_model():
        filename = consts.MODEL_FILE_PATH
        return pickle.load(open(filename, 'rb'))

    def predict(self, feature_vector: DMatrix):
        xgb_model = self.__instance.model
        return xgb_model.predict(feature_vector)

    def predict_probability(self, feature_vector: DMatrix):
        xgb_model = self.__instance.model
        return xgb_model.predict_proba(feature_vector)


if __name__ == '__main__':
    raw_json = """{"Current_Loan_Amount":445412.0,"Term":"Short_Term","Credit_Score":709.0,"Annual_Income":1167493.0,"Years_in_current_job":"8_years","Home_Ownership":"Home_Mortgage","Purpose":"Home_Improvements","Monthly_Debt":5214.74,"Years_of_Credit_History":17.2,"Months_since_last_delinquent":12,"Number_of_Open_Accounts":6.0,"Number_of_Credit_Problems":1.0,"Current_Credit_Balance":228190.0,"Maximum_Open_Credit":416746.0,"Bankruptcies":1.0,"Tax_Liens":0.0}"""
    json_data = json.loads(raw_json)

    row_to_predict = pd.read_json(path_or_buf=raw_json, orient="values", typ="series")
    row_to_predict_reshaped = row_to_predict.values.reshape(1, -1)
    # TODO - Add tests
    # TODO - read and apply cross validation and GridSearch
    model = Model()
    print(model.predict(row_to_predict_reshaped))
    print(model.predict_probability(row_to_predict_reshaped))





