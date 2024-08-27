# outOfOrderSeqNums() for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\outOfOrderSeqNums()\outOfOrderSeqNums().csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate the number of outOfOrderSeqNums
valueCounts = data.loc[(data['Suspected Out-of-Order'] == 1)]
print(valueCounts)
outOfOrderSeqNums = len(valueCounts)
print("The number of outOfOrderSeqNums frames .csv file is ", outOfOrderSeqNums)

# Count for loop to add a conditional counter towards calculation of the custom feature numOfSsdpFrames that will
# be used to calculate the custom feature numOfSsdpFramesPerSec().

count = 1

for index, row in data.iterrows():
    if row['Suspected Out-of-Order'] == 1:
        data.loc[index, 'outOfOrderSeqNums'] = count
        count += 1
    else:
        data.loc[index, 'outOfOrderSeqNums'] = count -1

data['outOfOrderSeqNums(Int)'] = data['outOfOrderSeqNums'].astype(int)

# Function that calculates the outOfOrderSeqNumsPerSec()
data['outOfOrderSeqNumsPerSec()'] = data['outOfOrderSeqNums'] / data['Time since reference or first frame']

new1 = data['outOfOrderSeqNumsPerSec()'].expanding().std()
data['outOfOrderSeqNumsPerSecStd'] = new1

# New outOfOrderSeqNumsPerSec Dataframe which gathers all the requested information towards the calculation of the
# outOfOrderSeqNumsPerSec().
outOfOrderSeqNumsPerSec = pd.DataFrame(data['Frame Type'])
outOfOrderSeqNumsPerSec['Sequence number'] = data['Sequence number']
outOfOrderSeqNumsPerSec['Time since reference or first frame'] = data['Time since reference or first frame']
outOfOrderSeqNumsPerSec['Suspected Out-of-Order'] = data['Suspected Out-of-Order']
outOfOrderSeqNumsPerSec['outOfOrderSeqNums'] = data['outOfOrderSeqNums']
outOfOrderSeqNumsPerSec['outOfOrderSeqNums(Int)'] = data['outOfOrderSeqNums(Int)']
outOfOrderSeqNumsPerSec['outOfOrderSeqNumsPerSec()'] = data['outOfOrderSeqNumsPerSec()']
outOfOrderSeqNumsPerSec['outOfOrderSeqNumsPerSecStd'] = data['outOfOrderSeqNumsPerSecStd']