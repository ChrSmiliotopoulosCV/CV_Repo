import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Importing the Dataset
# Read the original and labeled CSV file into the pandas 'df' dataframe
df1 = pd.read_csv(
    (
        r"C:\Users\chrsm\Desktop\Paper_no4 - Unsupervised ML LMD-2023 Paper\GitHub_Repository\testrepo\newAttempt(DeletedExtraRows).csv"
    ),
    encoding="utf-8",
    low_memory=False,
)

print(df1.head())

# df2 = df1.loc[df1['Label'] == 1.0]
df1['Label'] = df1['Label'].fillna(0).astype(int)


# print(df2['Label'].head())
# print(df2.head())
print(df1.dtypes)

df1.to_csv('newwwINT.csv')
