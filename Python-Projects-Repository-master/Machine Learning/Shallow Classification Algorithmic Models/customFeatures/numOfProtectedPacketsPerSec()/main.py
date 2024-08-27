# numOfProtectedPacketsPerSec() for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\numOfProtectedPacketsPerSec()\numOfProtectedPacketsPerSec().csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate Protected flag frames with type 'Data is protected'
valueCounts = data.loc[(data['Protected flag'] == 'Data is protected')]
print(valueCounts)
numOfProtectedPackets = len(valueCounts)
print("The number of Protected frames in .csv file is ", numOfProtectedPackets)

# Count for loop to add a conditional counter towards calculation of the custom feature numOfProtectedPackets that will
# be used to calculate the custom feature numOfProtectedPacketsPerSec().

count = 1

for index, row in data.iterrows():
    if row['Protected flag'] == 'Data is protected':
        data.loc[index, 'numOfProtectedPackets'] = count
        count += 1
    else:
        data.loc[index, 'numOfProtectedPackets'] = count -1

data['numOfProtectedPackets(Int)'] = data['numOfProtectedPackets'].astype(int)

# Function that calculates the numOfProtectedPacketsPerSec()
data['numOfProtectedPacketsPerSec()'] = data['numOfProtectedPackets'] / data['Time since reference or first frame']

# New numOfProtectedPacketsPerSec Dataframe which gathers all the requested information towards the calculation of the
# numOfProtectedPacketsPerSec().
numOfProtectedPacketsPerSec = pd.DataFrame(data['Frame Type'])
numOfProtectedPacketsPerSec['Time since reference or first frame'] = data['Time since reference or first frame']
numOfProtectedPacketsPerSec['Protected flag'] = data['Protected flag']
numOfProtectedPacketsPerSec['numOfProtectedPackets'] = data['numOfProtectedPackets']
numOfProtectedPacketsPerSec['numOfProtectedPackets(Int)'] = data['numOfProtectedPackets(Int)']
numOfProtectedPacketsPerSec['numOfProtectedPacketsPerSec()'] = data['numOfProtectedPacketsPerSec()']