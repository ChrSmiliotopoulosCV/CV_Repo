# Package Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Exploring the Dataset

df = pd.read_csv(
    r'D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Datasets\DATASET\DATASET\AWID-CLS-F-Tst\awid2Tst(all-without-scaling).csv', sep=',', error_bad_lines=False, encoding='ISO-8859-1', low_memory=False, verbose=True)

print("isnull() check for newAwid2df dataframe:")
print(df.isnull().any())
# newAwid2df.dropna() will remove the null, NaN and empty values from the dataframe.
df = df.dropna()
print("isnull() check for newAwid2df dataframe after dropna():")
print(df.isnull().any())
print(df['frame.len'].dtypes)
df['frame.len'] = pd.to_numeric(df['frame.len'], errors='coerce')
df = df.dropna()
print("isnull() check for newAwid2df dataframe after dropna():")
print(df.isnull().any())
print(df['frame.len'].dtypes)
# print("Number of null values in newAwid2df dataframe:")
# print(df.isnull().any().sum())
# df2 = pd.DataFrame(df.drop(df.columns[[16]], axis=1))
# #
# # print(df2.columns)
#
# # Create MinMaxScaler()
#
# minmaxscaler = MinMaxScaler()
#
# df3 = minmaxscaler.fit_transform(df2)
# df3 = pd.DataFrame(df3, columns=df2.columns)
# df3 = df3.join(df['class'])

#
# # Fit and transform in one step
#
# df3 = minmaxscaler.fit_transform(df2)
#
# # This statement guarantees that the name of each column stays the same after the transformation
#
# df3 = pd.DataFrame(df3, columns=df2.columns)
#
# # Adding the one hot encoded variables to the dataset
#
# minmax_df = pd.DataFrame(df.drop(df.columns[[0,1,2,3,4,5,6,7,8,9,10,11,12]], axis=1))
# minmax_df = minmax_df.join(df3)
#
# columns = minmax_df.columns
# columnIndex = columns.get_loc('wlan.fc.subtype')
# print(columnIndex)
#
# column_to_reorder = minmax_df.pop('Label')
# minmax_df.insert(13, 'Label', column_to_reorder)
#
# print(minmax_df.columns)

# Min-Max scaler
scaler = MinMaxScaler()
# All but Label into float

# df = df.astype(float)

# df = scaler.fit_transform(df)

# df['frame.len'] = df['frame.len'].convert_objects(convert_numeric=True)

# df[['frame.len', 'radiotap.length', 'radiotap.present.tsft', 'radiotap.channel.freq', 'radiotap.channel.type.cck',
#     'radiotap.channel.type.ofdm', 'radiotap.dbm_antsignal', 'wlan.fc.type', 'wlan.fc.subtype', 'wlan.fc.ds',
#     'wlan.fc.frag', 'wlan.fc.retry', 'wlan.fc.pwrmgt', 'wlan.fc.moredata', 'wlan.fc.protected',
#     'wlan.duration']] = df[['frame.len', 'radiotap.length', 'radiotap.present.tsft', 'radiotap.channel.freq', 'radiotap.channel.type.cck',
#     'radiotap.channel.type.ofdm', 'radiotap.dbm_antsignal', 'wlan.fc.type', 'wlan.fc.subtype', 'wlan.fc.ds',
#     'wlan.fc.frag', 'wlan.fc.retry', 'wlan.fc.pwrmgt', 'wlan.fc.moredata', 'wlan.fc.protected',
#     'wlan.duration']].astype(float)

df[['frame.len', 'radiotap.length', 'radiotap.present.tsft', 'radiotap.channel.freq', 'radiotap.channel.type.cck',
    'radiotap.channel.type.ofdm', 'radiotap.dbm_antsignal', 'wlan.fc.type', 'wlan.fc.subtype', 'wlan.fc.ds',
    'wlan.fc.frag', 'wlan.fc.retry', 'wlan.fc.pwrmgt', 'wlan.fc.moredata', 'wlan.fc.protected',
    'wlan.duration']] = scaler.fit_transform(
    df[['frame.len', 'radiotap.length', 'radiotap.present.tsft', 'radiotap.channel.freq', 'radiotap.channel.type.cck',
        'radiotap.channel.type.ofdm', 'radiotap.dbm_antsignal', 'wlan.fc.type', 'wlan.fc.subtype', 'wlan.fc.ds',
        'wlan.fc.frag', 'wlan.fc.retry', 'wlan.fc.pwrmgt', 'wlan.fc.moredata', 'wlan.fc.protected', 'wlan.duration']])

df.to_csv(
    r'D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Datasets\DATASET\DATASET\AWID-CLS-F-Trn\Preprocessed csv Files\awid2Tst(all-with-scaling)001.csv',
    index=False)
