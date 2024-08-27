# Import necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# Read CSV file

df = pd.read_csv(r'/Volumes/Backup_Volume/GitHub Repository/GitHub Python Projects/Python-Projects-Repository/Machine Learning/Update ML Scripts/pythonProject_MINMAXScaler/melb_data_numericalDraft.csv')

# Create MinMaxScaler()

scaler = MinMaxScaler()

# Fit and transform in one step

df2 = scaler.fit_transform(df)

# This statement guarantees that the name of each column stays the same after the transformation

df2 = pd.DataFrame(df2, columns=df.columns)

print(df2)