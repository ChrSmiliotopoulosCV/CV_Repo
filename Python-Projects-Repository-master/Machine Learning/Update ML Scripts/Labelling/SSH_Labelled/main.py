# SSH_Labelled pcap file, network traffic labelling with Python.
# Importing Libraries
import pandas as pd
import sklearn

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((r"C:\Users\chrsm\Desktop\csv files\SSH.csv"), encoding = "ISO-8859-1", low_memory=False)
data.head()

# for loop to add a conditional label in order to label the dataset's rows based on the conditions as "SSH" or "normal" traffic

for index, row in data.iterrows():
    if (1356015 <= row['Frame Number'] <= 2440390) and (row['IPv4 Address'] == '192.168.2.248'):
        data.loc[index, 'labelling'] = "SSH"
    else:
        data.loc[index, 'labelling'] = "normal"

data.to_csv(r"C:\Users\chrsm\Desktop\csv files\SSH_Labelled_Labelled.csv")



