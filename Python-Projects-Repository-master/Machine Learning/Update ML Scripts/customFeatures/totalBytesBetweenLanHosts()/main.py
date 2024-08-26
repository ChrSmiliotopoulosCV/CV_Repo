# totalBytesBetweenLanHosts for .csv frames counting python script
# Importing Libraries
import pandas as pd
import sklearn
import numpy as np
from ipaddress import ip_network, ip_address

# Importing the Dataset
# Read data from CSV file into pandas dataframe

data = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Python Data Encoding Scripts\customFeatures\totalBytesBetweenLanHosts()\totalBytesBetweenLanHosts().csv"))
data.head()

# for loop to print dataset's columnName
for (columnName, columnData) in data.iteritems():
    print('Column Name : ', columnName)

# For loop which iterates the Dataframe for the Sender IP addresses of the condition and the relevant Sender MAC
# addresses. For each iteration the function data.loc[index, 'bytesBetweenHosts'] = data.loc[index, 'Frame length on
# the wire'] returns the column bytesBetweenHosts and in the same way another function returns the receiverAddress.
count = 1
bytesBetweenHosts = 0
for index, row in data.iterrows():
    if (row['Sender IP address'] == '192.168.2.1' or row['Sender IP address'] == '192.168.2.73' \
            or row['Sender IP address'] == '192.168.2.125' or row['Sender IP address'] == '192.168.2.190' \
            or row['Sender IP address'] == '192.168.2.184' or row['Sender IP address'] == '192.168.2.254' \
            or row['Sender IP address'] == '192.168.2.41' or row['Sender IP address'] == '192.168.2.42' \
            or row['Sender IP address'] == '192.168.2.160' or row['Sender IP address'] == '192.168.2.19' \
            or row['Sender IP address'] == '192.168.2.247' or row['Sender IP address'] == '192.168.2.120' \
            or row['Sender IP address'] == '192.168.2.130' or row['Sender IP address'] == '192.168.2.239' \
            or row['Sender IP address'] == '192.168.2.248' or row['Sender IP address'] == '192.168.2.21' \
            or row['Sender IP address'] == '20.50.64.3') and (row['Sender MAC address'] == '0c:9d:92:54:fe:30' \
            or row['Sender MAC address'] == '0c:9d:92:54:fe:34' \
            or row['Sender MAC address'] == '50:3e:aa:e4:01:93' or row['Sender MAC address'] == '24:f5:a2:ea:86:c3' \
            or row['Sender MAC address'] == '50:3e:aa:e3:1f:be' or row['Sender MAC address'] == '00:c0:ca:a8:29:56' \
            or row['Sender MAC address'] == 'a4:b1:c1:91:4c:72' or row['Sender MAC address'] == '94:e9:79:82:c5:77' \
            or row['Sender MAC address'] == '00:c0:ca:a8:26:3e' or row['Sender MAC address'] == 'b4:ce:40:c8:33:81' \
            or row['Sender MAC address'] == '88:66:a5:55:a2:d4' or row['Sender MAC address'] == 'a4:08:ea:2a:9a:01' \
            or row['Sender MAC address'] == '00:0c:29:1d:70:f5' or row['Sender MAC address'] == '00:0c:29:cf:08:aa' \
            or row['Sender MAC address'] == '00:0c:29:ec:72:02' or row['Sender MAC address'] == '04:ed:33:e0:24:82' \
            or row['Sender MAC address'] == '74:d0:2b:7c:5a:5e' or row['Sender MAC address'] == 'N/A'):
        data.loc[index, 'bytesBetweenHosts'] = data.loc[index, 'Frame length on the wire']
        data.loc[index, 'receiverAddress'] = data.loc[index, 'Receiver address']
        data.loc[index, 'cCount'] = 1
        bytesBetweenHosts = data.loc[index, 'Frame length on the wire']
    # else:
    #     data.loc[index, 'bytesBetweenHosts'] = bytesBetweenHosts

# The functions that precede group the Dataframe by the receiverAddress column and sum with the cumsum() function the
# bytes sent for each receiverAddress group.
data['cumSum'] = data.groupby('receiverAddress').bytesBetweenHosts.cumsum()
data['cumSum'] = data['cumSum'].fillna(method='ffill')
data['cumSum'] = data['cumSum'].fillna(0)
data['cumSumInt'] = data['cumSum'].astype(int)
data['packetExchangeBtw2Hosts'] = data.groupby('receiverAddress').cCount.cumsum()
data['packetExchangeBtw2Hosts'] = data['packetExchangeBtw2Hosts'].fillna(method='ffill')
data['packetExchangeBtw2Hosts'] = data['packetExchangeBtw2Hosts'].fillna(0)
data['packetExchangeBtw2HostsInt'] = data['packetExchangeBtw2Hosts'].astype(int)

# New bytesBetweenHosts Dataframe which gathers all the requested information towards the calculation of the
# cumSum().
bytesBetweenHosts = pd.DataFrame(data['Sender IP address'])
bytesBetweenHosts['Sender MAC address'] = data['Sender MAC address']
bytesBetweenHosts['Frame length on the wire'] = data['Frame length on the wire']
bytesBetweenHosts['receiverAddress'] = data['receiverAddress']
bytesBetweenHosts['cumSum'] = data['cumSum']
bytesBetweenHosts['cumSumInt'] = data['cumSumInt']
bytesBetweenHosts['packetExchangeBtw2Hosts'] = data['packetExchangeBtw2Hosts']
bytesBetweenHosts['packetExchangeBtw2HostsInt'] = data['packetExchangeBtw2HostsInt']



