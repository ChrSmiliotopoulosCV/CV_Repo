# ssidRepPerSec() for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\ssidRepPerSec()\ssidRepPerSec().csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate the number of retransmitted Retransmited SSID
valueCounts = data.loc[(data['SSID'] == 'ASUS')]
print(valueCounts)
ssidRep = len(valueCounts)
print("The number of Retransmited SSID .csv file is ", ssidRep)

# Count for loop to add a conditional counter towards calculation of the custom feature ssidRep that will
# be used to calculate the custom feature ssidRepPerSec().

count = 1

for index, row in data.iterrows():
    if (row['SSID'] == 'ASUS'):
        data.loc[index, 'ssidRep'] = count
        count += 1
    else:
        data.loc[index, 'ssidRep'] = count -1

data['ssidRep(Int)'] = data['ssidRep'].astype(int)

# Function that calculates the ssidRepPerSec()
data['ssidRepPerSec()'] = data['ssidRep'] / data['Time since reference or first frame']

# New numOfUnprotectedPacketsPerSec Dataframe which gathers all the requested information towards the calculation of the
# ssidRepPerSec().
ssidRepPerSec = pd.DataFrame(data['Frame Type'])
ssidRepPerSec['Flag Retry'] = data['Flag Retry']
ssidRepPerSec['Time since reference or first frame'] = data['Time since reference or first frame']
ssidRepPerSec['ssidRep'] = data['ssidRep']
ssidRepPerSec['ssidRep(Int)'] = data['ssidRep(Int)']
ssidRepPerSec['ssidRepPerSec()'] = data['ssidRepPerSec()']