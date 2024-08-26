import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Importing the Dataset
# Read the original and labeled CSV file into the pandas 'df' dataframe
df1 = pd.read_csv(
    (
        r"C:\Users\chrsm\Desktop\Paper_no4 - Unsupervised ML LMD-2023 Paper\GitHub_Repository\testrepo\newAttempt.csv"
    ),
    encoding="utf-8",
    low_memory=False,
)

data = df1.drop(labels=range(2144008, 2144015), axis=0)

print(data.head())
print(data.tail())

data.to_csv('newAttempt(DeletedExtraRows).csv')