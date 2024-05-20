#  write your code here 

import pandas as pd

df_nan = pd.read_csv('data/dataset/input.txt')
initial_rows = df_nan.shape[0]
df_nan.dropna(inplace=True)
end_rows = df_nan.shape[0]

print(f"{initial_rows} {end_rows}")