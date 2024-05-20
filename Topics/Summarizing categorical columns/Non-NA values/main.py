#  write your code here 

import pandas as pd


df_rocks = pd.read_csv('data/dataset/input.txt')
print(df_rocks['labels'].count())