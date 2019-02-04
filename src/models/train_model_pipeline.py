import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from common import consts, utils

train_data = pd.read_csv(consts.DATA_FILE_PATH)

y = train_data.Loan_Status
X = train_data.drop(['Loan_Status'], axis=1)

# impute nan numbers
# remove outliers
X = X.fillna(X.mode().iloc[0])
y = y.map(dict(Fully_Paid=1, Charged_Off=0))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipeline = make_pipeline(OneHotEncoder(handle_unknown='ignore'),
                         SimpleImputer(),
                         XGBClassifier(max_depth=10, scale_pos_weight=2))
pipeline.fit(X_train, y_train)


#pickle.dump(pipeline, open(consts.MODEL_FILE_PATH, 'wb'))
predictions = pipeline.predict(X_test)

utils.report_metrics(y_test, predictions)
