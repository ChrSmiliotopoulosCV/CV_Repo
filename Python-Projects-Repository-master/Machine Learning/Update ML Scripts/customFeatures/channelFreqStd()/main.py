# signalStrengthStd for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\channelFreqStd()\channelFreqStd().csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to calculate the total Mean of the Signal Strength values
channelFreqStd = data['Channel Frequency'].std()
print("The standard deviation of the Channel's Frequencies is ", channelFreqStd)

new1 = data['Channel Frequency'].expanding().std()
data['signalStrengthStd'] = new1