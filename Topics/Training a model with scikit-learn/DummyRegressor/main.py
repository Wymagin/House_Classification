#  write your code here 

import pandas as pd
from sklearn.dummy import DummyRegressor

df = pd.read_csv("data/dataset/input.txt")

X, y = df[["X"]], df["y"]

dummy_regr = DummyRegressor(strategy="quantile", quantile=0.4)
dummy_regr.fit(X, y)
predictions = dummy_regr.predict(X)
rounded_first_prediction = round(predictions[0], 4)
print(rounded_first_prediction)
