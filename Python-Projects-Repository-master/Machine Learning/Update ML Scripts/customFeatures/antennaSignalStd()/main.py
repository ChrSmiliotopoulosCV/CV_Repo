# antennaSignalStrengthStd for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\antennaSignalStd()\antennaSignalStrengthStd.csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Function to calculate the total Mean of the Signal Strength values
antennaSignalStrengthStd = data['Antenna signal'].std()
print("The standard deviation of the Antenna signal is ", antennaSignalStrengthStd)

new1 = data['Antenna signal'].expanding().std()
data['antennaSignalStrengthStd'] = new1