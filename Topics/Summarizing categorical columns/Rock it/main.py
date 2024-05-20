#  write your code here 
import pandas as pd

rocking_frame = pd.read_csv("data/dataset/input.txt")
print(rocking_frame['labels'].value_counts()['R'])