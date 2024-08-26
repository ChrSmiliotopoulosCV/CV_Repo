# avgPacketSize for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\DeauthFrames\deauth_datasample.csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate Management frames with Subtype == 12
valueCounts = data.loc[(data['Frame Type'] == 'Management frame') & (data['Subtype'] == 12)]
print(valueCounts)
# value_counts() function to calculate the number of Deauthentication frames
deauthFrames = valueCounts['Subtype'].value_counts()
print(deauthFrames)
numOfDeauthFrames = len(valueCounts)
print("The number of Deauth frames in .csv file is ", numOfDeauthFrames)

# Count for loop to add a conditional counter towards calculation of the custom feature
count = 1

for index, row in data.iterrows():
    if row['Frame Type'] == 'Management frame' and row['Subtype'] == 12:
        data.loc[index, 'numOfDeauthFrames'] = count
        new1 = data.loc[index, 'Frame length on the wire']
        count += 1
    else:
        data.loc[index, 'numOfDeauthFrames'] = count -1

data['numOfDeauthFrames'] = data['numOfDeauthFrames'].astype(int)

# Statements to calculate the cumsum() of the packet size on the wire and the Average based on the specific row the
# data['lenghCumSum()'] and the data['Frame Number']
data['lenghCumSum()'] = data['Frame length on the wire'].cumsum()
data['avgPacketSize'] = data['lenghCumSum()'] / data['Frame Number']
data['avgPacketSizeInt'] = data['avgPacketSize'].astype(int)
# data.drop('Frame length on the wire', inplace=True, axis=1)



