# httpReqPerSec for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"/Users/christossmiliotopoulos/Desktop/myRepos/myPersonal_Git_Record/Python Data Encoding Scripts/customFeatures/httpReqPerSecStd/httpReqPerSecStd.csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate the number of dnsRetrasmPerSec
valueCounts = data.loc[(data['Request Method'] == 'M-SEARCH') | (data['Request Method'] == 'GET')]
print(valueCounts)
httpReq = len(valueCounts)
print("The number of httpReq .csv file is ", httpReq)

# Count for loop to add a conditional counter towards calculation of the custom feature ssidRep that will
# be used to calculate the custom feature ssidRepPerSec().

count = 1

for index, row in data.iterrows():
    if row['Request Method'] == 'M-SEARCH' or row['Request Method'] == 'GET':
        data.loc[index, 'httpReq'] = count
        count += 1
    else:
        data.loc[index, 'httpReq'] = count -1

data['httpReq(Int)'] = data['httpReq'].astype(int)

# Function that calculates the ssidRepPerSec()
data['httpReqPerSec()'] = data['httpReq'] / data['Time since reference or first frame']

new1 = data['httpReq'].expanding().std()
data['httpReqPerSecStd'] = new1

# New numOfUnprotectedPacketsPerSec Dataframe which gathers all the requested information towards the calculation of the
# ssidRepPerSec().
httpReq = pd.DataFrame(data['Frame Type'])
httpReq['Flag Retry'] = data['Flag Retry']
httpReq['Time since reference or first frame'] = data['Time since reference or first frame']
httpReq['httpReq'] = data['httpReq']
httpReq['httpReq(Int)'] = data['httpReq(Int)']
httpReq['httpReqPerSec()'] = data['httpReqPerSec()']
httpReq['httpReqPerSecStd'] = data['httpReqPerSecStd']