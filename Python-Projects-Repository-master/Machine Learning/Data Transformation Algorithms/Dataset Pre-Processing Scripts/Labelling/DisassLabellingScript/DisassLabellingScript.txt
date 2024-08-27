# Disass pcap file, network traffic labelling with Python.
# Importing Libraries
import pandas as pd
import sklearn

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((r"C:\Users\chrsm\Desktop\csv files\Disass.csv"), encoding = "ISO-8859-1", low_memory=False)
data.head()

# for loop to add a conditional label in order to label the dataset's rows based on the conditions as "disass" or "normal" traffic

for index, row in data.iterrows():
    if (row['Protected flag'] == 'Data is not protected') and (row['Frame Number'] >= 1404237 or row['Frame Number'] <= 2013346) and (row['Subtype'] == 10 or row['Subtype'] == 12):
        data.loc[index, 'labelling'] = "disass"
    else:
        data.loc[index, 'labelling'] = "normal"

data.to_csv(r"C:\Users\chrsm\Desktop\csv files\disass_Labelled.csv")



