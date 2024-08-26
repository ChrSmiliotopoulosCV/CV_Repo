# dnsRetrasmPerSec() for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\dnsRetrasmPerSec\dnsRetrasmPerSec.csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate the number of dnsRetrasmPerSec
valueCounts = data.loc[data['DNS Retransmission Flag'] == 1]
print(valueCounts)
dnsRetrasm1 = len(valueCounts)
print("The number of dnsRetrasm .csv file is ", dnsRetrasm1)

# Count for loop to add a conditional counter towards calculation of the custom feature ssidRep that will
# be used to calculate the custom feature ssidRepPerSec().

count = 1

for index, row in data.iterrows():
    if row['DNS Retransmission Flag'] == 1:
        data.loc[index, 'dnsRetrasm1'] = count
        count += 1
    else:
        data.loc[index, 'dnsRetrasm1'] = count -1

data['dnsRetrasm1(Int)'] = data['dnsRetrasm1'].astype(int)

# Function that calculates the ssidRepPerSec()
data['dnsRetrasmPerSec()'] = data['dnsRetrasm1'] / data['Time since reference or first frame']

# New numOfUnprotectedPacketsPerSec Dataframe which gathers all the requested information towards the calculation of the
# ssidRepPerSec().
dnsRetrasmPerSec = pd.DataFrame(data['Frame Type'])
dnsRetrasmPerSec['Flag Retry'] = data['Flag Retry']
dnsRetrasmPerSec['Time since reference or first frame'] = data['Time since reference or first frame']
dnsRetrasmPerSec['dnsRetrasm1'] = data['dnsRetrasm1']
dnsRetrasmPerSec['dnsRetrasm1(Int)'] = data['dnsRetrasm1(Int)']
dnsRetrasmPerSec['dnsRetrasmPerSec()'] = data['dnsRetrasmPerSec()']