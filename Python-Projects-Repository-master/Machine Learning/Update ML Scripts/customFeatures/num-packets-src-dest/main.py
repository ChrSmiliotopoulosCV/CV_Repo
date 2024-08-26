# num-packets-src-dest for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"/Users/christossmiliotopoulos/Desktop/myRepos/myPersonal_Git_Record/Python Data Encoding Scripts/customFeatures/num-packets-src-dest/num-packets-src-dest.csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate the number of num-packets-src-dest
valueCounts = data.loc[(data['DS status'] == 'Frame from DS to a STA via AP(To DS: 0 From DS: 1)')]
print(valueCounts)
dsStatus = len(valueCounts)
print("The number of Frames from DS to a STA via AP(To DS: 0 From DS: 1) .csv file is ", dsStatus)

# Count for loop to add a conditional counter towards calculation of the feature dsStatus that will
# be used to calculate the custom feature num-packets-src-dest and num-packets-src-dest per second.

count = 1

for index, row in data.iterrows():
    if row['DS status'] == 'Frame from DS to a STA via AP(To DS: 0 From DS: 1)':
        data.loc[index, 'num-packets-src-dest'] = count
        count += 1
    else:
        data.loc[index, 'num-packets-src-dest'] = count -1

data['num-packets-src-dest(Int)'] = data['num-packets-src-dest'].astype(int)

# Function that calculates the ssidRepPerSec()
data['num-packets-src-destPerSec()'] = data['num-packets-src-dest'] / data['Time since reference or first frame']

new1 = data['num-packets-src-destPerSec()'].expanding().std()
data['num-packets-src-destPerSecStd'] = new1

# New numOfUnprotectedPacketsPerSec Dataframe which gathers all the requested information towards the calculation of the
# ssidRepPerSec().
dsStatus= pd.DataFrame(data['DS status'])
# dsStatus.to_csv((
#     r"/Users/christossmiliotopoulos/Desktop/myRepos/myPersonal_Git_Record/Python Data Encoding Scripts/customFeatures/httpReqPerSecStd/httpReqPerSecStd001.csv"), index = False)
dsStatus['Time since reference or first frame'] = data['Time since reference or first frame']
dsStatus['num-packets-src-dest'] = data['num-packets-src-dest']
dsStatus['num-packets-src-dest(Int)'] = data['num-packets-src-dest(Int)']
dsStatus['num-packets-src-destPerSec()'] = data['num-packets-src-destPerSec()']
dsStatus['num-packets-src-destPerSecStd'] = data['num-packets-src-destPerSecStd']