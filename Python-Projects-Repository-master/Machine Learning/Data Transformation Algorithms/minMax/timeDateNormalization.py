# Normalize time series data
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
# load the dataset and print the first 5 rows
series = read_csv('daily-min-temperatures.csv', header=0, index_col=0)

print(type(series))

print(series.head())
# print(series)

# prepare data for normalization
values = series.values
print(type(values))
values = values.reshape((len(values), 1))
print(type(values))
print(values)

# train the normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
# normalize the dataset and print the first 5 rows
normalized = scaler.transform(values)
for i in range(5):
 print(normalized[i])

 # inverse transform and print the first 5 rows
inversed = scaler.inverse_transform(normalized)
for i in range(5):
 print(inversed[i])