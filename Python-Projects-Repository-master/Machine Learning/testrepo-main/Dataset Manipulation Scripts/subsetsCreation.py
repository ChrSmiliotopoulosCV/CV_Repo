# Importing Libraries
import pandas as pd
import sklearn
from tqdm import tqdm

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((r"C:\Users\chrsm\Desktop\LMD-2023\LMD-2023\LMD-2023 [2.3M Elements]\LMD-2023 [2.3M Elements]Checked\Labelled LMD-2023\LMD-2023 [2.3M Elements][Labelled]checked.csv"), encoding="ISO-8859-1", low_memory=False)
print(data.head())
# print(data)

print(len(data.index))
print(data.shape[0])
print(data.shape[1])

df = data[data['Label'] == 0] 
# print(df1)
print('df1 subset is comprised of ', len(df.index), 'rows labeled as "Normal [0]".')
df.to_csv('LMD-2023 [1.75M Elements - Normal]checked.csv')

df1 = data[data['Label'] == 1] 
# print(df1)
print('df1 subset is comprised of ', len(df1.index), 'rows labeled as "EoRS [1]".')
df1.to_csv('LMD-2023 [1.75M Elements - EoRS]checked.csv')

df2 = data[data['Label'] == 2] 
# print(df2)
print('df2 subset is comprised of ', len(df2.index), 'rows labeled as "EoHT [2]".')
df2.to_csv('LMD-2023 [1.75M Elements - EoHT]checked.csv')