# importing pandas
import pandas as pd
# importing numpy
import numpy as np
# importing OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# # reading dataset
# df = pd.read_csv(r'C:\Users\chrsm\Desktop\test.csv')
# # print (df)
# df.head()
#
# # checking features
# cat = df.select_dtypes(include='O').keys()
# # display variabels
# cat
# # creating new df
# # setting columns we use
new_df = pd.read_csv(r'C:\Users\chrsm\Desktop\test.csv', usecols=['Neighborhood'])
#new_df.head()

# unique values in each columns
for x in new_df.columns:
    #prinfting unique values
    print(x ,':', len(new_df[x].unique()))

# finding the top 20 categories
new_df.Neighborhood.value_counts().head(20)

# make list with top 10 variables
top_10 = [x for x in new_df.Neighborhood.value_counts().sort_values(ascending=False).head(10).index]
top_10

# make binary of labels
for Mitchel in top_10:
    new_df[Mitchel] = np.where(new_df['Neighborhood']==Mitchel,1,0)
    #new_df[['Neighborhood']+top_10]


