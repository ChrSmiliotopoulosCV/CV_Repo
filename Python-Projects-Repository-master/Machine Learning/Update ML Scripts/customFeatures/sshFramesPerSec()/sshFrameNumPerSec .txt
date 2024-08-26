# sshFramesPerSec() for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"C:\Users\chrsm\Desktop\ssh_selection2.csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate the number of sshFrameNum
valueCounts = data.loc[(data['Supported Protocol'] == 'SSH-2.0-libssh2_1.8.0') | (data['Supported Protocol'] == 'SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.1')]
print(valueCounts)
sshFrameNum = len(valueCounts)
print("The number of ssh frames .csv file is ", sshFrameNum)

# Count for loop to add a conditional counter towards calculation of the custom feature sshFrameNum that will
# be used to calculate the custom feature sshFrameNumPerSec().

count = 1

for index, row in data.iterrows():
    if row['Supported Protocol'] == 'SSH-2.0-libssh2_1.8.0' or row['Supported Protocol'] == 'SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.1':
        data.loc[index, 'sshFrameNum'] = count
        count += 1
    else:
        data.loc[index, 'sshFrameNum'] = count -1

data['sshFrameNum(Int)'] = data['sshFrameNum'].astype(int)

# Function that calculates the ssidRepPerSec()
data['sshFrameNumPerSec()'] = data['sshFrameNum'] / data['Time since reference or first frame']

new1 = data['sshFrameNumPerSec()'].expanding().std()
data['sshFrameNumPerSecStd'] = new1

# New numOfUnprotectedPacketsPerSec Dataframe which gathers all the requested information towards the calculation of the
# sshFrameNumPerSec().
sshFrameNumPerSec = pd.DataFrame(data['Frame Type'])
sshFrameNumPerSec['Supported Protocol'] = data['Supported Protocol']
sshFrameNumPerSec['Time since reference or first frame'] = data['Time since reference or first frame']
sshFrameNumPerSec['Subtype'] = data['Subtype']
sshFrameNumPerSec['sshFrameNum'] = data['sshFrameNum']
sshFrameNumPerSec['sshFrameNum(Int)'] = data['sshFrameNum(Int)']
sshFrameNumPerSec['sshFrameNumPerSec()'] = data['sshFrameNumPerSec()']
sshFrameNumPerSec['sshFrameNumPerSecStd'] = data['sshFrameNumPerSecStd']