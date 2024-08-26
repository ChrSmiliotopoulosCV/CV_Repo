# numOfSsdpFramesPerSec()() for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\numOfSsdpFramesPerSec()\numOfSsdpFramesPerSec().csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate the number of numOfSsdpFrames
valueCounts = data.loc[(data['Simple Service Discovery Protocol'] == 1)]
print(valueCounts)
numOfSsdpFrames = len(valueCounts)
print("The number of numOfReassocFrames frames .csv file is ", numOfSsdpFrames)

# Count for loop to add a conditional counter towards calculation of the custom feature numOfSsdpFrames that will
# be used to calculate the custom feature numOfSsdpFramesPerSec().

count = 1

for index, row in data.iterrows():
    if row['Simple Service Discovery Protocol'] == 1:
        data.loc[index, 'numOfSsdpFrames'] = count
        count += 1
    else:
        data.loc[index, 'numOfSsdpFrames'] = count -1

data['numOfSsdpFrames(Int)'] = data['numOfSsdpFrames'].astype(int)

# Function that calculates the ssidRepPerSec()
data['numOfSsdpFramesPerSec()'] = data['numOfSsdpFrames'] / data['Time since reference or first frame']

new1 = data['numOfSsdpFramesPerSec()'].expanding().std()
data['numOfSsdpFramesPerSecStd'] = new1

# New numOfReassocFramesPerSec Dataframe which gathers all the requested information towards the calculation of the
# numOfReassocFramesPerSec().
numOfReassocFramesPerSec = pd.DataFrame(data['Frame Type'])
# numOfAssocFramesPerSec['Supported Protocol'] = data['Supported Protocol']
# numOfAssocFramesPerSec['Time since reference or first frame'] = data['Time since reference or first frame']
numOfReassocFramesPerSec['Simple Service Discovery Protocol'] = data['Simple Service Discovery Protocol']
numOfReassocFramesPerSec['numOfSsdpFrames'] = data['numOfSsdpFrames']
numOfReassocFramesPerSec['numOfSsdpFrames(Int)'] = data['numOfSsdpFrames(Int)']
numOfReassocFramesPerSec['numOfSsdpFramesPerSec()'] = data['numOfSsdpFramesPerSec()']
numOfReassocFramesPerSec['numOfSsdpFramesPerSecStd'] = data['numOfSsdpFramesPerSecStd']