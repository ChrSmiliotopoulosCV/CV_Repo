# reasonCode3RetransPerSec() for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\reasonCode3RetransPerSec()\reasonCode3RetransPerSec().csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate the number of retransmitted EAPOL number 1 or 3 Messages
valueCounts = data.loc[(data['Reason code'] == 'Class 3 frame received from nonassociated STA') & (data['Flag Retry'] == 'Frame is being retransmitted')]
print(valueCounts)
reasonCode3RetransPerSec = len(valueCounts)
print("The number of Retransmited Reason Code 3 messages .csv file is ", reasonCode3RetransPerSec)

# Count for loop to add a conditional counter towards calculation of the custom feature numOfProtectedPackets that will
# be used to calculate the custom feature numOfProtectedPacketsPerSec().

count = 1

for index, row in data.iterrows():
    if (row['Reason code'] == 'Class 3 frame received from nonassociated STA') and (row['Flag Retry'] == 'Frame is being retransmitted'):
        data.loc[index, 'reasonCode3Retrans'] = count
        count += 1
    else:
        data.loc[index, 'reasonCode3Retrans'] = count -1

data['reasonCode3Retrans(Int)'] = data['reasonCode3Retrans'].astype(int)

# Function that calculates the numOfNum1_3RetransEAPOLMessagesPerSec()
data['reasonCode3RetransPerSec()'] = data['reasonCode3Retrans'] / data['Time since reference or first frame']

# New numOfUnprotectedPacketsPerSec Dataframe which gathers all the requested information towards the calculation of the
# numOfNum1_3RetransEAPOLMessagesPerSec().
reasonCode3RetransPerSec = pd.DataFrame(data['Frame Type'])
reasonCode3RetransPerSec['Flag Retry'] = data['Flag Retry']
reasonCode3RetransPerSec['Time since reference or first frame'] = data['Time since reference or first frame']
reasonCode3RetransPerSec['reasonCode3Retrans'] = data['reasonCode3Retrans']
reasonCode3RetransPerSec['reasonCode3Retrans(Int)'] = data['reasonCode3Retrans(Int)']
reasonCode3RetransPerSec['reasonCode3RetransPerSec()'] = data['reasonCode3RetransPerSec()']