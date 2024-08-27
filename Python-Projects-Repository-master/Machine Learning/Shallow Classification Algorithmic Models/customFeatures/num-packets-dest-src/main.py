# num-packets-dest-src for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"/Users/christossmiliotopoulos/Desktop/myRepos/myPersonal_Git_Record/Python Data Encoding Scripts/customFeatures/num-packets-dest-src/num-packets-dest-src.csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate the number of num-packets-dest-src
valueCounts = data.loc[(data['DS status'] == 'Frame from STA to DS via an AP (To DS: 1 From DS: 0)')]
print(valueCounts)
dsStatus = len(valueCounts)
print("The number of Frames from STA to DS via an AP (To DS: 1 From DS: 0) .csv file is ", dsStatus)

# Count for loop to add a conditional counter towards calculation of the feature dsStatus that will
# be used to calculate the custom feature num-packets-src-dest and num-packets-src-dest per second.

count = 1

for index, row in data.iterrows():
    if row['DS status'] == 'Frame from STA to DS via an AP (To DS: 1 From DS: 0)':
        data.loc[index, 'num-packets-dest-src'] = count
        count += 1
    else:
        data.loc[index, 'num-packets-dest-src'] = count -1

data['num-packets-dest-src(Int)'] = data['num-packets-dest-src'].astype(int)

# Function that calculates the ssidRepPerSec()
data['num-packets-dest-srcPerSec()'] = data['num-packets-dest-src'] / data['Time since reference or first frame']

new1 = data['num-packets-dest-srcPerSec()'].expanding().std()
data['num-packets-dest-srcPerSecStd'] = new1

# New numOfUnprotectedPacketsPerSec Dataframe which gathers all the requested information towards the calculation of the
# ssidRepPerSec().
dsStatus= pd.DataFrame(data['DS status'])
# dsStatus.to_csv((
#     r"/Users/christossmiliotopoulos/Desktop/myRepos/myPersonal_Git_Record/Python Data Encoding Scripts/customFeatures/httpReqPerSecStd/httpReqPerSecStd001.csv"), index = False)
dsStatus['Time since reference or first frame'] = data['Time since reference or first frame']
dsStatus['num-packets-dest-src'] = data['num-packets-dest-src']
dsStatus['num-packets-dest-src(Int)'] = data['num-packets-dest-src(Int)']
dsStatus['num-packets-dest-srcPerSec()'] = data['num-packets-dest-srcPerSec()']
dsStatus['num-packets-dest-srcPerSecStd'] = data['num-packets-dest-srcPerSecStd']
# dsStatus['num-packets-dest-srcPerSecStd'] = data['num-packets-dest-srcPerSecStd']
