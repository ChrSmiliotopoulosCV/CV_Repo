# importing packages 
import datetime 
import pandas as pd 
from datetime import datetime
import random

n_days = 428802

# today's date in timestamp 
# base = pd.Timestamp.today() 
timestamp0 = 1631000409
timestamp1 = 1665387333
timestamp2 = 1668071542
timestamp3 = 1700828084
timestamp4 = 1701587001

# #################################################
# #################### Round 0 ####################
# #################################################
dt_object0 = datetime.fromtimestamp(timestamp0)
print(dt_object0)

dataTime0 = [dt_object0]
print(dataTime0)

print(type(dt_object0))
# print(base)
# print(type(base))

import datetime 
# calculating timestamps for the next 10 days 
# timestamp_list = [dt_object + datetime.timedelta(seconds=x, milliseconds=x, microseconds=x) for x in range(n_days)] 
timestamp_list1 = [dt_object0 + datetime.timedelta(seconds=x+random.randint(1, 59), milliseconds=random.randint(1, 999), microseconds=random.randint(1, 999)) for x in range(n_days)] 

# iterating through timestamp_list 
for x in timestamp_list1: 
	print(x)
	dataTime0.append(x)

# print(dataTime)

df0 = pd.DataFrame(dataTime0, columns=['Timestamps'])
# print(df)

# ############################################################################################################################################################################################
# Attention!!! The following two lines of code were used unsuccessfully to manipulate the newly created timestamp towards the transformation of the timestamp's type() from ''datetime'' to ''string (str)''
# However, although that worked fine, when the .csv file was imported to the dfToOHEMinMax.py script for pre-processing the whole process failed to get fullfilled. It was found out that the 
# (str) type is not the ideal for datetime timestamps manipulation into features. 
# ############################################################################################################################################################################################

# df1['date1'] = df1.Timestamps.astype(str) + 'Ζ'
# df1 = df1.drop("Timestamps", axis='columns')

# print(df)

df0.to_csv('chris0.csv')

# #################################################
# #################### Round 1 ####################
# #################################################
from datetime import datetime
dt_object1 = datetime.fromtimestamp(timestamp1)
print(dt_object1)

dataTime1 = [dt_object1]
print(dataTime1)

print(type(dt_object1))
# print(base)
# print(type(base))

import datetime 
# calculating timestamps for the next 10 days 
# timestamp_list = [dt_object + datetime.timedelta(seconds=x, milliseconds=x, microseconds=x) for x in range(n_days)] 
timestamp_list1 = [dt_object1 + datetime.timedelta(seconds=x+random.randint(1, 59), milliseconds=random.randint(1, 999), microseconds=random.randint(1, 999)) for x in range(n_days)] 

# iterating through timestamp_list 
for x in timestamp_list1: 
	print(x)
	dataTime1.append(x)

# print(dataTime)

df1 = pd.DataFrame(dataTime1, columns=['Timestamps'])
# print(df)

# ############################################################################################################################################################################################
# Attention!!! The following two lines of code were used unsuccessfully to manipulate the newly created timestamp towards the transformation of the timestamp's type() from ''datetime'' to ''string (str)''
# However, although that worked fine, when the .csv file was imported to the dfToOHEMinMax.py script for pre-processing the whole process failed to get fullfilled. It was found out that the 
# (str) type is not the ideal for datetime timestamps manipulation into features. 
# ############################################################################################################################################################################################

# df1['date1'] = df1.Timestamps.astype(str) + 'Ζ'
# df1 = df1.drop("Timestamps", axis='columns')

# print(df)

df1.to_csv('chris1.csv')

# #################################################
# #################### Round 2 ####################
# #################################################
from datetime import datetime
dt_object2 = datetime.fromtimestamp(timestamp2)
print(dt_object2)

dataTime2 = [dt_object2]
print(dataTime2)

print(type(dt_object2))
# print(base)
# print(type(base))

import datetime 
# calculating timestamps for the next 10 days 
# timestamp_list = [dt_object + datetime.timedelta(seconds=x, milliseconds=x, microseconds=x) for x in range(n_days)] 
timestamp_list2 = [dt_object2 + datetime.timedelta(seconds=x+random.randint(1, 59), milliseconds=random.randint(1, 999), microseconds=random.randint(1, 999)) for x in range(n_days)] 

# iterating through timestamp_list 
for x in timestamp_list2: 
	print(x)
	dataTime2.append(x)

# print(dataTime)

df2 = pd.DataFrame(dataTime2, columns=['Timestamps'])
# print(df)

# ############################################################################################################################################################################################
# Attention!!! Read the comment in Roung 1.
# ############################################################################################################################################################################################

# df2['date1'] = df2.Timestamps.astype(str) + 'Ζ'
# df2 = df2.drop("Timestamps", axis='columns')

# print(df)

df2.to_csv('chris2.csv')

# #################################################
# #################### Round 3 ####################
# #################################################
from datetime import datetime
dt_object3 = datetime.fromtimestamp(timestamp3)
print(dt_object3)

dataTime3 = [dt_object3]
print(dataTime3)

print(type(dt_object3))
# print(base)
# print(type(base))

import datetime 
# calculating timestamps for the next 10 days 
# timestamp_list = [dt_object + datetime.timedelta(seconds=x, milliseconds=x, microseconds=x) for x in range(n_days)] 
timestamp_list3 = [dt_object3 + datetime.timedelta(seconds=x+random.randint(1, 59), milliseconds=random.randint(1, 999), microseconds=random.randint(1, 999)) for x in range(n_days)] 

# iterating through timestamp_list 
for x in timestamp_list3: 
	print(x)
	dataTime3.append(x)

# print(dataTime)

df3 = pd.DataFrame(dataTime3, columns=['Timestamps'])
# print(df)

# ############################################################################################################################################################################################
# Attention!!! Read the comment in Roung 1.
# ############################################################################################################################################################################################

# df3['date1'] = df3.Timestamps.astype(str) + 'Ζ'
# df3 = df3.drop("Timestamps", axis='columns')

# print(df)

df3.to_csv('chris3.csv')

# #################################################
# #################### Round 4 ####################
# #################################################
from datetime import datetime
dt_object4 = datetime.fromtimestamp(timestamp4)
print(dt_object4)

dataTime4 = [dt_object4]
print(dataTime4)

print(type(dt_object4))
# print(base)
# print(type(base))

import datetime 
# calculating timestamps for the next 10 days 
# timestamp_list = [dt_object + datetime.timedelta(seconds=x, milliseconds=x, microseconds=x) for x in range(n_days)] 
timestamp_list4 = [dt_object4 + datetime.timedelta(seconds=x+random.randint(1, 59), milliseconds=random.randint(1, 999), microseconds=random.randint(1, 999)) for x in range(n_days)] 

# iterating through timestamp_list 
for x in timestamp_list4: 
	print(x)
	dataTime4.append(x)

# print(dataTime)

df4 = pd.DataFrame(dataTime4, columns=['Timestamps'])
# print(df)

# ############################################################################################################################################################################################
# Attention!!! Read the comment in Roung 1.
# ############################################################################################################################################################################################

# df4['date1'] = df4.Timestamps.astype(str) + 'Ζ'
# df4 = df4.drop("Timestamps", axis='columns')

# print(df)

df4.to_csv('chris4.csv')