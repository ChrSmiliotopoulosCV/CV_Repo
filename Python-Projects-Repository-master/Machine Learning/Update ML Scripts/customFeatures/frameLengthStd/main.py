# Association request frames counting python script
# Importing Libraries
import pandas
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\frameLengthStd\frameLengthStd.csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to locate the condition to locate number of input numbers
numOfInputs = len(data)
print("The number of input frames in .csv file is ", numOfInputs)
# Function to calculate the total Standard Deviation of the Frame length on the wire
stdFrameLength = data['Frame length on the wire'].std(axis=0)
print("The standard deviation of the Frame length on the wire is ", stdFrameLength)

# Function to calculate the cumulative or expanding Standard Deviation of the Frame length on the wire and add the
# result to a new column on the same Dataframe
new = data['Frame length on the wire'].rolling(len(data), min_periods=2).std()
data['Std_Cumulative'] = new

new1 = data['Frame length on the wire'].expanding().std()
data['Std_Expanding()'] = new1

# new2 = data['Frame length on the wire'].rolling(2).std()
# data['Std_Rolling()'] = new2
