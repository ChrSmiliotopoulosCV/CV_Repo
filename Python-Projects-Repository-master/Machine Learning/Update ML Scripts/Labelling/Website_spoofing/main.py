# Website_Spoofing_Labelled pcap file, network traffic labelling with Python.
# Importing Libraries
import pandas as pd
import sklearn

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((r"C:\Users\chrsm\Desktop\csv files\Website_Spoofing.csv"), encoding="ISO-8859-1", low_memory=False)
data.head()

# for loop to add a conditional label in order to label the dataset's rows based on the conditions as "Website_Spoofing" or
# "normal" traffic

for index, row in data.iterrows():
    if (16410 <= row['Frame Number'] <= 2668583) and ((row['Source address'] == '04:ed:33:e0:24:82')
                                                      or (row['Destination address'] == '04:ed:33:e0:24:82')
                                                      or (row['Source address'] == '00:C0:CA:A8:29:56')
                                                      or (row['Destination address'] == '00:C0:CA:A8:29:56')
                                                      or (row['Source address'] == '24:F5:A2:EA:86:C3')
                                                      or (row['Destination address'] == '24:F5:A2:EA:86:C3')
                                                      or (row['Source address'] == '00:C0:CA:A8:26:3E')
                                                      or (row['Destination address'] == '00:C0:CA:A8:26:3E')):
        data.loc[index, 'labelling'] = "Website_Spoofing"
    else:
        data.loc[index, 'labelling'] = "normal"

data.to_csv(r"C:\Users\chrsm\Desktop\csv files\Website_Spoofing_Labelled.csv")
