# Deauthentication frames counting python script
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

# data['numOfDeauthFrames'] = data.where((data['Frame Type'] == 'Management frame') & (data['Subtype'] == 12)).groupby('Frame Number').ngroup()+1
# data.drop('Frame length on the wire', inplace=True, axis=1)

# for ind, row in data.iterrows():
#     if row['Frame Type'] == 'Management frame' and row['Subtype'] == 12:
#         data.loc[ind, 'newNum'] = count
#         count += 1
#     else:
#         data.loc[ind, 'newNum'] = count -1
#
# data.drop('Frame length on the wire', inplace=True, axis=1)

# Count for loop to add a conditional counter towards calculation of the custom feature
count = 1

for index, row in data.iterrows():
    if row['Frame Type'] == 'Management frame' and row['Subtype'] == 12:
        data.loc[index, 'numOfDeauthFrames'] = count
        count += 1
    else:
        data.loc[index, 'numOfDeauthFrames'] = count -1
data['numOfDeauthFrames'] = data['numOfDeauthFrames'].astype(int)
data.drop('Frame length on the wire', inplace=True, axis=1)



