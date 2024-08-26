# Package Imports
import pandas as pd

# Exploring the Dataset
df = pd.read_csv(r'C:\Users\chrsm\Desktop\melb_data.csv')
df.head()

# Identify only the Categorical Variables from the Dataset
s = (df.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)

# Create a new Dataset with three selected categorical columns
features = df[['Type', 'Method', 'Regionname']]
features.head()

# Function which prints the Method distinct values counting
print(features.Method.value_counts())

# One hot encoding with get_dummies function the selected column
df2 = pd.get_dummies(features['Type'])
df2.value_counts()


