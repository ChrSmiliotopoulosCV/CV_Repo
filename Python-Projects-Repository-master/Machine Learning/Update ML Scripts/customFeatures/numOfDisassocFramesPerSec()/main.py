# numOfDisassocFramesPerSec() for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\numOfDisassocFramesPerSec()\numOfDisassocFramesPerSec().csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate the number of numOfDisassocFrames
valueCounts = data.loc[(data['Frame Type'] == 'Management frame') & (data['Subtype'] == 10)]
print(valueCounts)
numOfDisassocFrames = len(valueCounts)
print("The number of numOfDisassocFrames frames .csv file is ", numOfDisassocFrames)

# Count for loop to add a conditional counter towards calculation of the custom feature numOfDisassocFrames that will
# be used to calculate the custom feature numOfDisassocFramesPerSec().

count = 1

for index, row in data.iterrows():
    if row['Frame Type'] == 'Management frame' and row['Subtype'] == 10:
        data.loc[index, 'numOfDisassocFrames'] = count
        count += 1
    else:
        data.loc[index, 'numOfDisassocFrames'] = count -1

data['numOfDisassocFrames(Int)'] = data['numOfDisassocFrames'].astype(int)

# Function that calculates the ssidRepPerSec()
data['numOfDisassocFramesPerSec()'] = data['numOfDisassocFrames'] / data['Time since reference or first frame']

new1 = data['numOfDisassocFramesPerSec()'].expanding().std()
data['numOfDisassocFramesPerSecStd'] = new1

# New numOfDisassocFramesPerSec Dataframe which gathers all the requested information towards the calculation of the
# numOfDisassocFramesPerSec().
numOfDisassocFramesPerSec = pd.DataFrame(data['Frame Type'])
# numOfAssocFramesPerSec['Supported Protocol'] = data['Supported Protocol']
# numOfAssocFramesPerSec['Time since reference or first frame'] = data['Time since reference or first frame']
numOfDisassocFramesPerSec['Subtype'] = data['Subtype']
numOfDisassocFramesPerSec['numOfDisassocFrames'] = data['numOfDisassocFrames']
numOfDisassocFramesPerSec['numOfDisassocFrames(Int)'] = data['numOfDisassocFrames(Int)']
numOfDisassocFramesPerSec['numOfDisassocFramesPerSec()'] = data['numOfDisassocFramesPerSec()']
numOfDisassocFramesPerSec['numOfDisassocFramesPerSecStd'] = data['numOfDisassocFramesPerSecStd']