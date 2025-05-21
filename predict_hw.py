import pandas as pd
import numpy as np
import catboost
from catboost import CatBoostRegressor, Pool

data = pd.read_csv("mtcars.csv")

label = data['mpg']

train1 = data.drop(['mpg'],axis=1)

col_imp = ['model', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am','gear', 'carb']
cat_features = ['model']

clf = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    cat_features=cat_features,
    verbose=100
)
clf.fit(train1, label)


def predict(dict_values, col_imp=col_imp, clf=clf):
    x = pd.DataFrame([{col: dict_values[col] for col in col_imp}])
    y_pred = clf.predict(x)[0]
    return y_pred