# signalStrengthMean for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\signalStrengthMean()\signalStrengthMean.csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to calculate the total Mean of the Signal Strength values
signalStrengthMean = data['Signal strength ()'].mean()
print("The mean() of the Antenna signal is ", signalStrengthMean)

new1 = data['Signal strength ()'].expanding().mean()
data['signalStrengthMean'] = new1