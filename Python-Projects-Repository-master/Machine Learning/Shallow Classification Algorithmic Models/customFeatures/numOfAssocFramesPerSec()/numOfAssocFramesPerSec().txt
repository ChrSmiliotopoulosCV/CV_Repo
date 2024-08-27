# numOfAssocFramesPerSec() for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\numOfAssocFramesPerSec()\numOfAssocFramesPerSec().csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate the number of numOfAssocFrames
valueCounts = data.loc[(data['Frame Type'] == 'Management frame') & (data['Subtype'] == 0)]
print(valueCounts)
numOfAssocFrames = len(valueCounts)
print("The number of ssh frames .csv file is ", numOfAssocFrames)

# Count for loop to add a conditional counter towards calculation of the custom feature numOfAssocFrames that will
# be used to calculate the custom feature numOfAssocFramesPerSec().

count = 1

for index, row in data.iterrows():
    if row['Frame Type'] == 'Management frame' and row['Subtype'] == 0:
        data.loc[index, 'numOfAssocFrames'] = count
        count += 1
    else:
        data.loc[index, 'numOfAssocFrames'] = count -1

data['numOfAssocFrames(Int)'] = data['numOfAssocFrames'].astype(int)

# Function that calculates the ssidRepPerSec()
data['numOfAssocFramesPerSec()'] = data['numOfAssocFrames'] / data['Time since reference or first frame']

new1 = data['numOfAssocFramesPerSec()'].expanding().std()
data['numOfAssocFramesPerSecStd'] = new1

# New numOfAssocFramesPerSec Dataframe which gathers all the requested information towards the calculation of the
# numOfAssocFramesPerSec().
numOfAssocFramesPerSec = pd.DataFrame(data['Frame Type'])
# numOfAssocFramesPerSec['Supported Protocol'] = data['Supported Protocol']
# numOfAssocFramesPerSec['Time since reference or first frame'] = data['Time since reference or first frame']
numOfAssocFramesPerSec['Subtype'] = data['Subtype']
numOfAssocFramesPerSec['numOfAssocFrames'] = data['numOfAssocFrames']
numOfAssocFramesPerSec['numOfAssocFrames(Int)'] = data['numOfAssocFrames(Int)']
numOfAssocFramesPerSec['numOfAssocFramesPerSec()'] = data['numOfAssocFramesPerSec()']
numOfAssocFramesPerSec['numOfAssocFramesPerSecStd'] = data['numOfAssocFramesPerSecStd']