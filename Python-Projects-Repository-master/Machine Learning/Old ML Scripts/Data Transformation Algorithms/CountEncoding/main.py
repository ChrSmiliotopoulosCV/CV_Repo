# Package Imports
import columns as columns
import ks as ks
import pandas as pd
import train as train

from sklearn.preprocessing import OneHotEncoder

import category_encoders as ce

# Dataset Exploration

df = pd.read_csv(r'C:\Users\chrsm\Desktop\melb_data02.csv')
df.head()

# Get the categorical data out of training data and print the list. The object dtype indicates a column has text.

s = (df.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)

# For simplicity, I’ve taken up only 3 categorical columns to illustrate encoding techniques.

features = df[['Type', 'Method', 'Regionname']]
features.head()

# For loop to find the different counts of categories in a dict and apply them to the variable count_encoded
# which will be joined with the original df Dataframe

for i in range(0,3):
    count_enc = ce.CountEncoder()
    cat_features = features
    count_encoded = count_enc.fit_transform(cat_features)
    data = (df.join(count_encoded.add_suffix("_count")))
