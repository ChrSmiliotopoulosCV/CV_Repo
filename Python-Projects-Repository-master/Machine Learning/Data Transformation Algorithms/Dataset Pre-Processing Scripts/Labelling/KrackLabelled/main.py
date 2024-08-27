# Krack pcap file, network traffic labelling with Python.
# Importing Libraries
import pandas as pd
import sklearn

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((r"C:\Users\chrsm\Desktop\csv files\Krack.csv"), encoding = "ISO-8859-1", low_memory=False)
data.head()

# for loop to add a conditional label in order to label the dataset's rows based on the conditions as "Krack" or "normal" traffic

for index, row in data.iterrows():
    if (row['Channel Num'] == 2):
        data.loc[index, 'labelling'] = "Krack"
    else:
        data.loc[index, 'labelling'] = "normal"

data.to_csv(r"C:\Users\chrsm\Desktop\csv files\Krack_Labelled.csv")



