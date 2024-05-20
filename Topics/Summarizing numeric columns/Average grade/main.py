# Your code here. The DataFrame is already loaded as grades
grades = grades.drop('Student', axis = 1)
print(grades.mean(axis = 1))
