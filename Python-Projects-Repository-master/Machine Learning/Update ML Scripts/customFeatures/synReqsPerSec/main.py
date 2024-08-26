# synReqsPerSec for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\httpReqPerSecStd\httpReqPerSecStd.csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate the number of synReqsPerSec
valueCounts = data.loc[(data['SYN Flag'] == 'Set')]
print(valueCounts)
synReqs = len(valueCounts)
print("The number of synReqsPerSec .csv file is ", synReqs)

# Count for loop to add a conditional counter towards calculation of the custom feature ssidRep that will
# be used to calculate the custom feature ssidRepPerSec().

count = 1

for index, row in data.iterrows():
    if row['SYN Flag'] == 'Set':
        data.loc[index, 'synReqs'] = count
        count += 1
    else:
        data.loc[index, 'synReqs'] = count -1

data['synReqs(Int)'] = data['synReqs'].astype(int)

# Function that calculates the ssidRepPerSec()
data['synReqsPerSec'] = data['synReqs'] / data['Time since reference or first frame']

new1 = data['synReqsPerSec'].expanding().std()
data['synReqsPerSecStd'] = new1

# New numOfUnprotectedPacketsPerSec Dataframe which gathers all the requested information towards the calculation of the
# ssidRepPerSec().
synReqs = pd.DataFrame(data['Frame Type'])
synReqs['Flag Retry'] = data['Flag Retry']
synReqs['Time since reference or first frame'] = data['Time since reference or first frame']
synReqs['SYN Flag'] = data['SYN Flag']
synReqs['synReqs'] = data['synReqs']
synReqs['synReqs(Int)'] = data['synReqs(Int)']
synReqs['synReqsPerSec'] = data['synReqsPerSec']
synReqs['synReqsPerSecStd'] = data['synReqsPerSecStd']