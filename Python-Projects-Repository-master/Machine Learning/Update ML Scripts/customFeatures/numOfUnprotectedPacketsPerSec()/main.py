# numOfUnprotectedPacketsPerSec() for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\numOfUnprotectedPacketsPerSec()\numOfUnprotectedPacketsPerSec().csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate Protected flag frames with type 'Data is protected'
valueCounts = data.loc[(data['Protected flag'] == 'Data is not protected')]
print(valueCounts)
numOfUnprotectedPackets = len(valueCounts)
print("The number of Protected frames in .csv file is ", numOfUnprotectedPackets)

# Count for loop to add a conditional counter towards calculation of the custom feature numOfProtectedPackets that will
# be used to calculate the custom feature numOfProtectedPacketsPerSec().

count = 1

for index, row in data.iterrows():
    if row['Protected flag'] == 'Data is not protected':
        data.loc[index, 'numOfUnprotectedPackets'] = count
        count += 1
    else:
        data.loc[index, 'numOfUnprotectedPackets'] = count -1

data['numOfUnprotectedPackets(Int)'] = data['numOfUnprotectedPackets'].astype(int)

# Function that calculates the numOfUnprotectedPacketsPerSec()
data['numOfUnprotectedPacketsPerSec()'] = data['numOfUnprotectedPackets'] / data['Time since reference or first frame']

# New numOfUnprotectedPacketsPerSec Dataframe which gathers all the requested information towards the calculation of the
# numOfProtectedPacketsPerSec().
numOfUnprotectedPacketsPerSec = pd.DataFrame(data['Frame Type'])
numOfUnprotectedPacketsPerSec['Time since reference or first frame'] = data['Time since reference or first frame']
numOfUnprotectedPacketsPerSec['Protected flag'] = data['Protected flag']
numOfUnprotectedPacketsPerSec['numOfUnprotectedPackets'] = data['numOfUnprotectedPackets']
numOfUnprotectedPacketsPerSec['numOfUnprotectedPackets(Int)'] = data['numOfUnprotectedPackets(Int)']
numOfUnprotectedPacketsPerSec['numOfUnprotectedPacketsPerSec()'] = data['numOfUnprotectedPacketsPerSec()']