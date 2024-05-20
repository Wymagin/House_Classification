#  write your code here 

import pandas as pd


input_text = "data/dataset/input.txt"
rock_df = pd.read_csv(input_text)
median_M = round(rock_df.loc[rock_df['labels'] == 'M', 'null_deg'].median(), 3)
median_R = round(rock_df.loc[rock_df['labels'] == 'R', 'null_deg'].median(), 3)
print("M = {} R = {}".format(median_M, median_R))