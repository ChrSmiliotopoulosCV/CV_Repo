# Data Frames frames counting python script
# Importing Libraries
import pandas as pd
import sklearn

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\numOfDataFrames\numOfDataFrames.csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate Management frames with Subtype == 12
valueCounts = data.loc[(data['Frame Type'] == 'Data frame')]
print(valueCounts)
# value_counts() function to calculate the number of Data frames
dataFrames = valueCounts['Frame Type'].value_counts()
print(dataFrames)
numOfDataFrames = len(valueCounts)
print("The number of Data frames in .csv file is ", numOfDataFrames)

# data['numOfDataFrames'] = data.where((data['Frame Type'] == 'Data frame')).groupby('Frame Number').ngroup()+1

# Count for loop to add a conditional counter towards calculation of the custom feature
count = 1

for index, row in data.iterrows():
    if row['Frame Type'] == 'Data frame':
        data.loc[index, 'numOfDataFrames'] = count
        count += 1
    else:
        data.loc[index, 'numOfDataFrames'] = count -1
data['numOfDataFrames'] = data['numOfDataFrames'].astype(int)
data.drop('Frame length on the wire', inplace=True, axis=1)