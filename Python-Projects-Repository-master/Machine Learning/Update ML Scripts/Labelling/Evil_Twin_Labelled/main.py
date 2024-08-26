# Evil_Twin_Labelled pcap file, network traffic labelling with Python.
# Importing Libraries
import pandas as pd
import sklearn

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((r"C:\Users\chrsm\Desktop\csv files\Malware.csv"), encoding="ISO-8859-1", low_memory=False)
data.head()

# for loop to add a conditional label in order to label the dataset's rows based on the conditions as "Evil_Twin" or
# "normal" traffic

for index, row in data.iterrows():
    if (1420038 <= row['Frame Number'] <= 3778728) and\
            ((row['Subtype'] == 8) or (row['Subtype'] == 10) or (row['Subtype'] == 12) or (row['Subtype'] == 40)) and\
            (row['Frame length on the wire'] < 242) and (row['Protected flag'] == 'Data is not protected') and\
            (row['IPv4 Address'] == '192.168.30.1') or (row['Destination address'] == '0c:9d:92:54:fe:35'):
        data.loc[index, 'labelling'] = "Evil_Twin"
    else:
        data.loc[index, 'labelling'] = "normal"

data.to_csv(r"C:\Users\chrsm\Desktop\csv files\Evil_Twin_Labelled.csv")
