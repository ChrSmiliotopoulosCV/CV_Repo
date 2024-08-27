# numOfAckFlags() for .csv frames counting python script
# Importing Libraries
from idlelib.iomenu import encoding

import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\numOfAckFlags()\Krack.csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate numOfAckFlags() == Set
valueCounts = data.loc[(data['Acknowledgment Flag'] == 'Set')]
print(valueCounts)
# value_counts() function to calculate the number of numOfAckFlags() == Set
ackFlagFrames = valueCounts['Acknowledgment Flag'].value_counts()
print(ackFlagFrames)
numOfAckFlagFrames = len(valueCounts)
print("The number of Deauth frames in .csv file is ", numOfAckFlagFrames)

# Count for loop to add a conditional counter towards calculation of the custom feature numOfAckFlags()
count = 1

for index, row in data.iterrows():
    if row['Acknowledgment Flag'] == 'Set':
        data.loc[index, 'numOfAckFlags()'] = count
        new1 = data.loc[index, 'Acknowledgment Flag']
        count += 1
    else:
        data.loc[index, 'numOfAckFlags()'] = count -1

data['numOfAckFlags(Int)'] = data['numOfAckFlags()'].astype(int)

# Python statements to create an new Dataframe with all the relevant information for the calculation of the
# numOfAckFlags() custom feature.
numOfAckFlags = pd.DataFrame(data['Frame Number'])
numOfAckFlags['Acknowledgment Flag'] = data['Acknowledgment Flag']
numOfAckFlags['Time since reference or first frame'] = data['Time since reference or first frame']
numOfAckFlags['numOfAckFlags()'] = data['numOfAckFlags()']
numOfAckFlags['numOfAckFlags(Int)'] = data['numOfAckFlags(Int)']



