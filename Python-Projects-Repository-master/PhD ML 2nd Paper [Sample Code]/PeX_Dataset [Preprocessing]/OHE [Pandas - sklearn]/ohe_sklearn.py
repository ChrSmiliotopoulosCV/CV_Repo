# Package Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

# Exploring the Dataset

df = pd.read_csv(r'/Volumes/Backup_Volume/GitHub Repository/GitHub Python Projects/Python-Projects-Repository/PhD ML 2nd Paper [Sample Code]/PeX_Dataset [Preprocessing]/melb_data.csv')

# Converting the type column to categorical

method_df = pd.DataFrame(df[['Type']])
method_df['Type'] = method_df['Type'].astype('category')

# One hot encoding the selected column

enc_df = enc.fit_transform(df.Method.values.reshape(-1, 1)).toarray()

# Adding the one hot encoded variables to the dataset

ohe_variable = pd.DataFrame(enc_df, columns=["encType_" + str(int(i)) for i in range(enc_df.shape[1])])

method_df = method_df.join(ohe_variable)
method_df
print(method_df)

# SOS SOS SOS we should look the following link
# https://blog.cambridgespark.com/robust-one-hot-encoding-in-python-3e29bfcec77e in order to fix the auto selection
# of the variable name during OHE procedure
