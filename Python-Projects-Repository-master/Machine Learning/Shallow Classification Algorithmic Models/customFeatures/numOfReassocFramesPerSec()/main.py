# numOfReassocFramesPerSec() for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\numOfReassocFramesPerSec()\numOfReassocFramesPerSec().csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate the number of numOfReassocFrames
valueCounts = data.loc[(data['Frame Type'] == 'Management frame') & (data['Subtype'] == 2)]
print(valueCounts)
numOfReassocFrames = len(valueCounts)
print("The number of numOfReassocFrames frames .csv file is ", numOfReassocFrames)

# Count for loop to add a conditional counter towards calculation of the custom feature numOfReassocFrames that will
# be used to calculate the custom feature numOfReassocFramesPerSec().

count = 1

for index, row in data.iterrows():
    if row['Frame Type'] == 'Management frame' and row['Subtype'] == 2:
        data.loc[index, 'numOfReassocFrames'] = count
        count += 1
    else:
        data.loc[index, 'numOfReassocFrames'] = count -1

data['numOfReassocFrames(Int)'] = data['numOfReassocFrames'].astype(int)

# Function that calculates the ssidRepPerSec()
data['numOfReassocFramesPerSec()'] = data['numOfReassocFrames'] / data['Time since reference or first frame']

new1 = data['numOfReassocFramesPerSec()'].expanding().std()
data['numOfReassocFramesPerSecStd'] = new1

# New numOfReassocFramesPerSec Dataframe which gathers all the requested information towards the calculation of the
# numOfReassocFramesPerSec().
numOfReassocFramesPerSec = pd.DataFrame(data['Frame Type'])
# numOfAssocFramesPerSec['Supported Protocol'] = data['Supported Protocol']
# numOfAssocFramesPerSec['Time since reference or first frame'] = data['Time since reference or first frame']
numOfReassocFramesPerSec['Subtype'] = data['Subtype']
numOfReassocFramesPerSec['numOfReassocFrames'] = data['numOfReassocFrames']
numOfReassocFramesPerSec['numOfReassocFrames(Int)'] = data['numOfReassocFrames(Int)']
numOfReassocFramesPerSec['numOfReassocFramesPerSec()'] = data['numOfReassocFramesPerSec()']
numOfReassocFramesPerSec['numOfReassocFramesPerSecStd'] = data['numOfReassocFramesPerSecStd']