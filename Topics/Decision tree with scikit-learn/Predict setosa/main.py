#  write your code here

import pandas as pd

from sklearn.metrics import classification_report

true_labels = ['apple', 'banana', 'orange', 'pear', 'apple', 'banana', 'orange', 'pear']
predicted_labels = ['apple', 'banana', 'orange', 'pear', 'orange', 'banana', 'orange', 'apple']

print(classification_report(true_labels, predicted_labels, output_dict = True))


# from sklearn.datasets import load_iris
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
#
# X, y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# classifier = DecisionTreeClassifier(random_state=42, max_depth=3)
#
# classifier.fit(X_train, y_train)
# score = classifier.score(X_test, y_test)
# print(score)
# score = classifier.score(X_train, y_train)
# print(score)





# df = pd.read_csv('data/dataset/input.txt')
# X, y = load_iris(return_X_y=True)
# classifier = DecisionTreeClassifier(random_state=42)
# classifier.fit(X, y)
# pred_train = classifier.predict(df)
# prediction = (pred_train == 0).sum()
# print(prediction)
