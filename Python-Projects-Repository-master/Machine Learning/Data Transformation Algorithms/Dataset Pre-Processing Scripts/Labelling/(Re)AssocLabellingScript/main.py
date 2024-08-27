# (Re)AssocLabellingScript pcap file, network traffic labelling with Python.
# Importing Libraries
import pandas as pd
import sklearn

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((r"C:\Users\chrsm\Desktop\csv files\(Re)Assoc.csv"), encoding = "ISO-8859-1", low_memory=False)
data.head()

# for loop to add a conditional label in order to label the dataset's rows based on the conditions as "(Re)Assoc" or "normal" traffic

for index, row in data.iterrows():
    if (row['Frame Number'] >= 1145178  or row['Frame Number'] <= 1833964) and (row['Subtype'] == 0 or row['Subtype'] == 0 or row['Subtype'] == 8):
        data.loc[index, 'labelling'] = "(Re)Assoc"
    else:
        data.loc[index, 'labelling'] = "normal"

data.to_csv(r"C:\Users\chrsm\Desktop\csv files\(Re)Assoc_Labelled.csv")



