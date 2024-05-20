import os
import requests
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from category_encoders import TargetEncoder


if __name__ == '__main__':
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")
house_df = pd.read_csv('../Data/house_class.csv')
X = house_df[['Area', 'Room', 'Lon', 'Lat', 'Zip_area', 'Zip_loc']]
y = house_df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=X['Zip_loc'])
categorical_features = ['Zip_area', 'Zip_loc', 'Room']
X_train_ordinal = X_train.copy()
X_test_ordinal = X_test.copy()
X_train_target = X_train.copy()
X_test_target = X_test.copy()
# OneHotEncoder


ohencoder = OneHotEncoder(drop='first')
ohencoder.fit(X_train[categorical_features])

X_train_transformed = pd.DataFrame(ohencoder.transform(X_train[categorical_features]).toarray(), index=X_train.index)
X_test_transformed = pd.DataFrame(ohencoder.transform(X_test[categorical_features]).toarray(), index=X_test.index)

X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)
X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)

X_train_final.columns = X_train_final.columns.astype(str)
X_test_final.columns = X_test_final.columns.astype(str)

dtc = DecisionTreeClassifier(criterion='entropy',
                             max_features=3,
                             splitter='best',
                             max_depth=6,
                             min_samples_split=4,
                             random_state=3)
dtc.fit(X_train_final, y_train)

y_pred = dtc.predict(X_test_final)
accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)
# print(round(accuracy, 4))
# print(X_train_final)

# OrdinalEncoder

ordencoder = OrdinalEncoder()
ordencoder.fit(X_train_ordinal[categorical_features])

# to jest jakos zle niby ohe tylko tak dziala

# X_train_transformed = pd.DataFrame(ordencoder.transform(X_train_ordinal[categorical_features]), index=X_train.index)
# X_test_transformed = pd.DataFrame(ordencoder.transform(X_test_ordinal[categorical_features]), index=X_test.index)
# to jest niby git

X_train_ordinal[categorical_features] = ordencoder.transform(X_train_ordinal[categorical_features])
X_test_ordinal[categorical_features] = ordencoder.transform(X_test_ordinal[categorical_features])
# print(X_train_ordinal.columns)
# Niby dziala dlatego ze ['Area', 'Lon', 'Lat'] zostaly wywolane osobno
X_train_ordinal_final = X_train_ordinal[['Area', 'Lon', 'Lat']].join(X_train_ordinal[categorical_features])
X_test_ordinal_final = X_test_ordinal[['Area', 'Lon', 'Lat']].join(X_test_ordinal[categorical_features])

X_train_ordinal_final.columns = X_train_ordinal_final.columns.astype(str)
X_test_ordinal_final.columns = X_test_ordinal_final.columns.astype(str)

dtcord = DecisionTreeClassifier(criterion='entropy',
                                max_features=3,
                                splitter='best',
                                max_depth=6,
                                min_samples_split=4,
                                random_state=3)
dtcord.fit(X_train_ordinal_final, y_train)

y_pred_ordinal = dtcord.predict(X_test_ordinal_final)
accuracyord = accuracy_score(y_test, y_pred_ordinal)
# print(round(accuracyord, 4))
#
# # Target encoder
# print(X_train_target)
# to not see warnings about future changes
pd.set_option('future.no_silent_downcasting', True)

# invoke and fit encoder

tencoder = TargetEncoder(cols=['Room', 'Zip_area', 'Zip_loc'])
tencoder.fit(X_train_target[['Room', 'Zip_area', 'Zip_loc']], y_train)

# transform the ['Room', 'Zip_area', 'Zip_loc'] columns

X_train_target_transformed = tencoder.transform(X_train_target[['Room', 'Zip_area', 'Zip_loc']])
X_test_target_transformed = tencoder.transform(X_test_target[['Room', 'Zip_area', 'Zip_loc']])

# join transformed columns with rest

X_train_target_final = X_train_target[['Area', 'Lon', 'Lat']].join(X_train_target_transformed)
X_test_target_final = X_test_target[['Area', 'Lon', 'Lat']].join(X_test_target_transformed)

# initialize and fit dtclassifier

dtarget = DecisionTreeClassifier(criterion='entropy',
                                 max_features=3,
                                 splitter='best',
                                 max_depth=6,
                                 min_samples_split=4,
                                 random_state=3)
dtarget.fit(X_train_target_final, y_train)

# make prediction and calculate accuracy

y_pred_target = dtarget.predict(X_test_target_final)
accuracytarget = accuracy_score(y_test, y_pred_target)
# print(round(accuracytarget, 4))

# Report of the tree classifier models trained with different encoders

report_onehot = classification_report(y_test, y_pred, output_dict=True)
report_ordinal = classification_report(y_test, y_pred_ordinal, output_dict=True)
report_target = classification_report(y_test, y_pred_target, output_dict=True)
print(f"OneHotEncoder:{round(report_onehot['macro avg']['f1-score'], 2)}")
print(f"OrdinalEncoder:{round(report_ordinal['macro avg']['f1-score'], 2)}")
print(f"TargetEncoder:{round(round(report_target['macro avg']['f1-score'], 4), 3)}")
# print(X_train_target_final)
#
# print(X_train_target_transformed)

# print(ordenc.categories_)
# print(y_pred)
# print(X_train_final)
# print(encoder)
# print(encoder.categories_)
# print(X_train_encoded)
# print("------------------------- ")
# print(X_test_encoded)
# print(X_train_transformed)
# transformed = encoder.fit_transform(X[['Room'], ['Zip_area'], ['Zip_loc']])

# encoder.categories_[['Room'], ['Zip_area'], ['Zip_loc']]
# print(encoder)


# print(X.head())
# print(y.head())
# print(X_train['Zip_loc'].value_counts().to_dict())
