#  write your code here 

import pandas as pd


df_nan = pd.read_csv('data/dataset/input.txt')
nan_proportion = df_nan.isna().sum() / df_nan.shape[0]
print(round(nan_proportion, 2))
