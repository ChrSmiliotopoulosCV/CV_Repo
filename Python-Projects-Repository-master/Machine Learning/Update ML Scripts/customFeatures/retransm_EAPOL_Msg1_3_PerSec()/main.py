# retransm_EAPOL_Msg1_3_PerSec() for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\retransm_EAPOL_Msg1_3_PerSec()\retransm_EAPOL_Msg1_3_PerSec().csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate the number of retransmitted EAPOL number 1 or 3 Messages
valueCounts = data.loc[((data['Message number'] == 1) | (data['Message number'] == 3)) & (data['Flag Retry'] == 'Frame is being retransmitted')]
print(valueCounts)
numOfNum1_3RetransEAPOLMessages = len(valueCounts)
print("The number of Retransmited EAPOL 1 or 3 Messages in .csv file is ", numOfNum1_3RetransEAPOLMessages)

# Count for loop to add a conditional counter towards calculation of the custom feature numOfProtectedPackets that will
# be used to calculate the custom feature numOfProtectedPacketsPerSec().

count = 1

for index, row in data.iterrows():
    if (row['Message number'] == 1 or row['Message number'] == 3) and (row['Flag Retry'] == 'Frame is being retransmitted'):
        data.loc[index, 'numOfNum1_3RetransEAPOLMessages'] = count
        count += 1
    else:
        data.loc[index, 'numOfNum1_3RetransEAPOLMessages'] = count -1

data['numOfNum1_3RetransEAPOLMessages(Int)'] = data['numOfNum1_3RetransEAPOLMessages'].astype(int)

# Function that calculates the numOfNum1_3RetransEAPOLMessagesPerSec()
data['numOfNum1_3RetransEAPOLMessagesPerSec()'] = data['numOfNum1_3RetransEAPOLMessages'] / data['Time since reference or first frame']

# New numOfUnprotectedPacketsPerSec Dataframe which gathers all the requested information towards the calculation of the
# numOfNum1_3RetransEAPOLMessagesPerSec().
numOfNum1_3RetransEAPOLMessagesPerSec = pd.DataFrame(data['Frame Type'])
numOfNum1_3RetransEAPOLMessagesPerSec['Time since reference or first frame'] = data['Time since reference or first frame']
numOfNum1_3RetransEAPOLMessagesPerSec['Message number'] = data['Message number']
numOfNum1_3RetransEAPOLMessagesPerSec['Flag Retry'] = data['Flag Retry']
numOfNum1_3RetransEAPOLMessagesPerSec['numOfNum1_3RetransEAPOLMessages'] = data['numOfNum1_3RetransEAPOLMessages']
numOfNum1_3RetransEAPOLMessagesPerSec['numOfNum1_3RetransEAPOLMessages(Int)'] = data['numOfNum1_3RetransEAPOLMessages(Int)']
numOfNum1_3RetransEAPOLMessagesPerSec['numOfNum1_3RetransEAPOLMessagesPerSec()'] = data['numOfNum1_3RetransEAPOLMessagesPerSec()']