


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)














# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# link = "http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
# df_sonar = pd.read_csv(link, header=None)
# df_sonar[60] = df_sonar[60].replace(['M','R'], [0, 1],)
# # print(df_sonar.head())
# X = df_sonar.drop(60, axis=1)
# y = df_sonar[60]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=34)
# # print(df_sonar.head())
# clf = DecisionTreeClassifier(criterion='entropy', random_state=34, max_depth=7)
# clf = clf.fit(X_train, y_train)
# pred_train = clf.predict(X_train)
# prediction = pred_train[33]
# result = y_train.iloc[33]
# print(prediction, result)
# train_score = clf.score(X_train, y_train)
# test_score = clf.score(X_test, y_test)
# print("Accuracy on train set: {}".format(train_score))
# print("Accuracy on test set: {}".format(round(test_score, 3)))
# print(clf.get_depth())
# real_prices = np.array([1, 4, 3, 2])
# predicted_prices = np.array([0.9, 4.1, 2.9, 2.1])
#
# residuals = real_prices - predicted_prices
#
# squared_residuals = residuals ** 2
# mean_squared_error = np.mean(squared_residuals)
# rmse = np.sqrt(mean_squared_error)
# mean_target = np.mean(real_prices)
# print(mean_target)
# if mean_target != 0:
#     nrmse = rmse / mean_target
# else:
#     nrmse = rmse
# nrmse_rounded = round(nrmse, 2)