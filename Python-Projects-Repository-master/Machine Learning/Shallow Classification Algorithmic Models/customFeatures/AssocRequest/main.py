# Association request frames counting python script
# Importing Libraries
import pandas as pd
import sklearn

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\AssocRequest\assoc_request_dataSample.csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate Management frames with Subtype == 0
valueCounts = data.loc[(data['Frame Type'] == 'Management frame') & (data['Subtype'] == 0) & (data['Flag Retry'] == 'Frame is being retransmitted')]
print(valueCounts)
# value_counts() function to calculate the number of Assoc Request frames
assocReqFrames = valueCounts['Subtype'].value_counts()
print(assocReqFrames)
numOfAssocReqFrames = len(valueCounts)
print("The number of Assoc Request frames retransmitted in .csv file is ", numOfAssocReqFrames)

# data['numOfAssocReqFrames'] = data.where((data['Frame Type'] == 'Management frame') & (data['Subtype'] == 0) & (data['Flag Retry'] == 'Frame is being retransmitted')).groupby('Frame Number').ngroup()+1

# Count for loop to add a conditional counter towards calculation of the custom feature
count = 1

for index, row in data.iterrows():
    if row['Frame Type'] == 'Management frame' and row['Flag Retry'] == 'Frame is being retransmitted' and row['Subtype'] == 0:
        data.loc[index, 'numOfAssocReqFrames'] = count
        count += 1
    else:
        data.loc[index, 'numOfAssocReqFrames'] = count -1
data['numOfAssocReqFrames'] = data['numOfAssocReqFrames'].astype(int)
# data.drop('Frame length on the wire', inplace=True, axis=1)