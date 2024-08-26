# Kr00k pcap file, network traffic labelling with Python.
# Importing Libraries
import pandas as pd
import sklearn

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((r"C:\Users\chrsm\Desktop\csv files\Kr00k.csv"), encoding = "ISO-8859-1", low_memory=False)
data.head()

# for loop to add a conditional label in order to label the dataset's rows based on the conditions as "Kr00k" or "normal" traffic

for index, row in data.iterrows():
    if (row['Frame Number'] >= 1555898) and (row['Subtype'] == 10) and (row['Protected flag'] == 'Data is not protected'):
        data.loc[index, 'labelling'] = "Kr00k"
    else:
        data.loc[index, 'labelling'] = "normal"

data.to_csv(r"C:\Users\chrsm\Desktop\csv files\Kr00k_Labelled.csv")



