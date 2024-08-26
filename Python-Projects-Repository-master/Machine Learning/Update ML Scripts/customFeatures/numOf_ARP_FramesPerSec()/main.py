# numOf_ARP_FramesPerSec() for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\numOf_ARP_FramesPerSec()\numOf_ARP_FramesPerSec().csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate the number of numOf_ARP_Frames
valueCounts = data.loc[(data['Address Resolution Protocol'] == 1)]
print(valueCounts)
numOf_ARP_Frames = len(valueCounts)
print("The number of Address Resolution Protocol frames .csv file is ", numOf_ARP_Frames)

# Count for loop to add a conditional counter towards calculation of the custom feature numOf_ARP_Frames that will
# be used to calculate the custom feature numOf_ARP_FramesPerSec().

count = 1

for index, row in data.iterrows():
    if row['Address Resolution Protocol'] == 1:
        data.loc[index, 'numOf_ARP_Frames'] = count
        count += 1
    else:
        data.loc[index, 'numOf_ARP_Frames'] = count -1

data['numOf_ARP_Frames(Int)'] = data['numOf_ARP_Frames'].astype(int)

# Function that calculates the numOf_ARP_FramesPerSec()
data['numOf_ARP_FramesPerSec()'] = data['numOf_ARP_Frames'] / data['Time since reference or first frame']

new1 = data['numOf_ARP_FramesPerSec()'].expanding().std()
data['numOf_ARP_FramesPerSecStd'] = new1

# New numOf_ARP_FramesPerSec Dataframe which gathers all the requested information towards the calculation of the
# numOf_ARP_FramesPerSec().
numOf_ARP_FramesPerSec = pd.DataFrame(data['Frame Type'])
# numOfAssocFramesPerSec['Supported Protocol'] = data['Supported Protocol']
# numOfAssocFramesPerSec['Time since reference or first frame'] = data['Time since reference or first frame']
numOf_ARP_FramesPerSec['Address Resolution Protocol'] = data['Address Resolution Protocol']
numOf_ARP_FramesPerSec['numOf_ARP_Frames'] = data['numOf_ARP_Frames']
numOf_ARP_FramesPerSec['numOf_ARP_Frames(Int)'] = data['numOf_ARP_Frames(Int)']
numOf_ARP_FramesPerSec['numOf_ARP_FramesPerSec()'] = data['numOf_ARP_FramesPerSec()']
numOf_ARP_FramesPerSec['numOf_ARP_FramesPerSecStd'] = data['numOf_ARP_FramesPerSecStd']