import re
import pandas as pd

# my_dict = {"Name":[],"Address":[],"Age":[]};

# my_dict["Name"].append("Guru")
# my_dict["Name"].append("Christos")
# my_dict["Name"].append("Maria")
# my_dict["Name"].append("Kleo")
# my_dict["Name"].append("George")
# my_dict["Name"].append("Maria")
# my_dict["Address"].append("Mumbai")
# my_dict["Address"].append("Mumbai1")
# my_dict["Address"].append("Mumbai2")
# my_dict["Address"].append("Mumbai3")
# my_dict["Address"].append("Mumbai4")
# my_dict["Address"].append("Mumbai5")
# my_dict["Age"].append(30)
# my_dict["Age"].append(31)
# my_dict["Age"].append(32)
# my_dict["Age"].append(33)
# my_dict["Age"].append(34)	
# my_dict["Age"].append(35)
# print(my_dict)



# def add_value(dict_obj, key, value):
#     ''' Adds a key-value pair to the dictionary.
#         If the key already exists in the dictionary, 
#         it will associate multiple values with that 
#         key instead of overwritting its value'''
#     if key not in dict_obj:
#         dict_obj[key] = value
#     elif isinstance(dict_obj[key], list):
#         dict_obj[key].append(value)
#     else:
#         dict_obj[key] = [dict_obj[key], value]
# # Dictionary of names and phone numbers
# phone_details = {   'Mathew': 212323,
#                     'Ritika': 334455,
#                     'John'  : 345323 }
# # Append a value to the existing key 
# add_value(phone_details, 'John', 111223)
# # Append a value to the existing key
# add_value(phone_details, 'John', 333444)
# for key, value in phone_details.items():
#     print(key, ' - ', value)

# df = pd.DataFrame(phone_details)
# print(df)

df1 = pd.read_csv('df1.csv')
print(df1)

df2 = pd.read_csv('df2.csv')
print(df2)

result = pd.concat([df1, df2], ignore_index=True, sort=False)
print(result)
# result.to_csv('concatenated.csv')

result1 = pd.concat([df1,df2]).drop_duplicates().reset_index(drop=True)
print(result1)