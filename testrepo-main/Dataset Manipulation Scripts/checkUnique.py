import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Importing the Dataset
# Read the original and labeled CSV file into the pandas 'df' dataframe
df = pd.read_csv(
    (
        r"C:\Users\chrsm\Desktop\LMD-2023\LMD-2023\LMD-2023 [1.75M Elements]\LMD-2023 [1.75M Elements][Labelled]checked.csv"
    ),
    encoding="utf-8",
    low_memory=False,
)

print(df.head())
print(df['SystemTime'].nunique())