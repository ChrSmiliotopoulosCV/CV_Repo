# stdPayloadPacketLength() for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\stdPayloadPacketLength()\stdPayloadPacketLength().csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# Statements to calculate the cumsum() of the packet size on the wire and the Average based on the specific row the
# data['lenghCumSum()'] and the data['Frame Number']
data['lenghCumSum()'] = data['Frame length on the wire'].cumsum()
data['avgPacketSize'] = data['lenghCumSum()'] / data['Frame Number']
data['avgPacketSizeInt'] = data['avgPacketSize'].astype(int)

# Function to calculate the total Standard Deviation of the Frame length on the wire
stdPayloadPacketLength = data['TCP Segment Len'].std(axis=0)
print("The standard deviation of the TCP Segment Len is ", stdPayloadPacketLength)

# The sumPayloadPacketLength() .cumsum() calculation and avgPayloadPacketLength(), avgPayloadPacketLength(Int)
# calculations.
data['sumPayloadPacketLength()'] = data['TCP Segment Len'].cumsum()
data['avgPayloadPacketLength()'] = data['sumPayloadPacketLength()'] / data['Frame Number']
data['avgPayloadPacketLength(Int)'] = data['avgPayloadPacketLength()'].astype(int)
new1 = data['TCP Segment Len'].expanding().std()
data['stdPayloadPacketLength()'] = new1

avgPayloadPacketLength = pd.DataFrame(data['Frame Number'])
avgPayloadPacketLength['TCP Segment Len'] = data['TCP Segment Len']
avgPayloadPacketLength['sumPayloadPacketLength()'] = data['sumPayloadPacketLength()']
avgPayloadPacketLength['avgPayloadPacketLength()'] = data['avgPayloadPacketLength()']
avgPayloadPacketLength['avgPayloadPacketLength(Int)'] = data['avgPayloadPacketLength(Int)']
avgPayloadPacketLength['stdPayloadPacketLength()'] = data['stdPayloadPacketLength()']
