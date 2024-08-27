# bssidRepPerSec() for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\bssidRepPerSec()\bssidRepPerSec().csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate the number of retransmitted Retransmited SSID
valueCounts = data.loc[(data['BSS Id'] == '0c:9d:92:54:fe:30') & (data['Flag Retry'] == 'Frame is being retransmitted')]
print(valueCounts)
bssidRep = len(valueCounts)
print("The number of Retransmited BSSID .csv file is ", bssidRep)

# Count for loop to add a conditional counter towards calculation of the custom feature ssidRep that will
# be used to calculate the custom feature ssidRepPerSec().

count = 1

for index, row in data.iterrows():
    if row['BSS Id'] == '0c:9d:92:54:fe:30' and row['Flag Retry'] == 'Frame is being retransmitted':
        data.loc[index, 'bssidRep'] = count
        count += 1
    else:
        data.loc[index, 'bssidRep'] = count -1

data['bssidRep(Int)'] = data['bssidRep'].astype(int)

# Function that calculates the ssidRepPerSec()
data['bssidRepPerSec()'] = data['bssidRep'] / data['Time since reference or first frame']

# New numOfUnprotectedPacketsPerSec Dataframe which gathers all the requested information towards the calculation of the
# ssidRepPerSec().
bssidRepPerSec = pd.DataFrame(data['Frame Type'])
bssidRepPerSec['Flag Retry'] = data['Flag Retry']
bssidRepPerSec['Time since reference or first frame'] = data['Time since reference or first frame']
bssidRepPerSec['bssidRep'] = data['bssidRep']
bssidRepPerSec['bssidRep(Int)'] = data['bssidRep(Int)']
bssidRepPerSec['bssidRepPerSec()'] = data['bssidRepPerSec()']