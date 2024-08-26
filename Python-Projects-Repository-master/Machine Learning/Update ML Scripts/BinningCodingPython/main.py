import pandas as pd
import numpy as np
import seaborn as sns

raw_df = pd.read_csv(r'C:\Users\chrsm\Desktop\Binning_SampleData.csv')
df = raw_df.groupby(['name'])['ext price'].sum().reset_index()

df['ext price'].plot(kind='hist')

print(df['ext price'].describe())

print(pd.qcut(df['ext price'], q=10))

df['quantile_ex_1'] = pd.qcut(df['ext price'], q=10)
#df['quantile_ex_2'] = pd.qcut(df['ext price'], q=10, precision=0)

df.head()