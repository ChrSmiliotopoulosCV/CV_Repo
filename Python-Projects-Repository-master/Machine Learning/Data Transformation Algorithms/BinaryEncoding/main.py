# Package Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# Creating instance of one-hot-encoder
lb_style = LabelBinarizer()

# Exploring the Dataset

df = pd.read_csv(r'C:\Users\chrsm\Desktop\melb_data.csv')

# Converting the type column to categorical

method_df = pd.DataFrame(df[['Type']])
method_df['Type'] = method_df['Type'].astype('category')

# Binary encoding the selected column

lb_results = lb_style.fit_transform(method_df['Type'])

#binary_variable = pd.DataFrame(lb_results, columns=["binMethod_" + str(int(i)) for i in range(lb_results.shape[1])])

print(pd.DataFrame(lb_results, columns=lb_style.classes_).value_counts())

# Adding the binary encoded variables to the dataset

binary_df = pd.DataFrame(lb_results, columns= lb_style.classes_)