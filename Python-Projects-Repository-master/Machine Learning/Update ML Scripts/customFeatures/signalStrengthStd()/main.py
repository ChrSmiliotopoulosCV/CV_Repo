# signalStrengthStd for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\signalStrengthStd()\signalStrengthStd.csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to calculate the total Mean of the Signal Strength values
signalStrengthStd = data['Signal strength ()'].std()
print("The standard deviation of the Signal Strength is ", signalStrengthStd)

new1 = data['Signal strength ()'].expanding().std()
data['signalStrengthStd'] = new1