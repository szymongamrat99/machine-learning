import pandas as pd
import numpy as np
import math as mt
import random
import itertools
from sklearn import preprocessing as pre
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

filename = 'Churn_Modelling.csv'

df = pd.read_csv(filename, delimiter=";", decimal='.')

col_name = df.columns

for col in col_name:
  if col == "RowNumber":
    col_name = col_name.drop(col)
    df.pop(col)
  elif col == "CustomerId":
    col_name = col_name.drop(col)
    df.pop(col)
  elif col == "Surname":
    col_name = col_name.drop(col)
    df.pop(col)

# Features

target = ["Exited"]
features = col_name.drop(target)

nonNumFeat = ["Gender", "Geography"]
numFeat = features.drop(nonNumFeat)

features = ["NumOfProducts", "Age", "IsActiveMember", "Gender"]

all_permutations = list(itertools.permutations(features))

features = list(all_permutations[16])

le = pre.LabelEncoder()

for feat in nonNumFeat:
  df[feat] = le.fit_transform(df[feat])

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

clf = RandomForestClassifier(1000,criterion="entropy",random_state=42)

model = clf.fit(X_train, y_train.values.ravel())

import metrics as mt
import matplotlib.pyplot as plt

mt.FeatureImportance(model, features)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

mt.AccuracyScore(y_test, y_pred)
mt.ROCCurve(y_test, y_proba)

plt.show()

