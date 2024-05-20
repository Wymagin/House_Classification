#  write your code here 

from sklearn.dummy import DummyRegressor
from sklearn import datasets

dummy_regressor = DummyRegressor(strategy='median')
X, y = datasets.fetch_california_housing(return_X_y=True)
dummy_regressor.fit(X, y)
print(dummy_regressor)
print(X)
print(y)
print(dummy_regressor.constant_)
print(dummy_regressor.n_outputs_)
print(dummy_regressor.predict(X))







# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_wine
#
# data = load_wine(as_frame=True)["frame"]
# X, y = data.iloc[:, :-1], data["target"]
#
# print(data)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.86, random_state=100)
# print(X_test['ash'].sum())
# # import pandas as pd
# #
#
# df_nanns = pd.read_csv('data/dataset/input.txt')
#
# print(df_nanns.isna().any().sum())