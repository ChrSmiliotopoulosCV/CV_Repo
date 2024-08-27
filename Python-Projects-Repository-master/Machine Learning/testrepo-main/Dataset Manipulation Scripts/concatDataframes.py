import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Importing the Dataset
# Read the original and labeled CSV file into the pandas 'df' dataframe
df1 = pd.read_csv(
    (
        r"C:\Users\chrsm\Desktop\LMD-2023\LMD-2023\LMD-2023 [2.3M Elements]\OldFiles\LMD-2023 [2.3M Elements]\Labelled LMD-2023\LMD-2023 [2.3M Elements][Labelled].csv"
    ),
    encoding="utf-8",
    low_memory=False,
)

print(df1.head())

# df2 = pd.read_csv(
#     (
#         r"C:\Users\chrsm\Desktop\LMD-2023\LMD-2023\LMD-2023 [1.75M Elements]\LMD-2023 [1.75M Elements] Checked\Labelled LMD-2023\dfLMD23Label.csv"
#     ),
#     encoding="utf-8",
#     low_memory=False,
# )

df3 = pd.read_csv(
    (
        r"C:\Users\chrsm\Desktop\LMD-2023\LMD-2023\LMD-2023 [2.3M Elements]\OldFiles\LMD-2023 [2.3M Elements]\Labelled LMD-2023\concatedChrisFinalcls.csv"
    ),
    encoding="utf-8",
    low_memory=False,
)

print(df3.head())

result = pd.concat([df1, df3], axis=1)
print(result.head())

result.to_csv('newAttempt.csv')