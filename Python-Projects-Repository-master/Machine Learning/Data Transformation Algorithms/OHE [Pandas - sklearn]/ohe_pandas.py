# Loading a Sample DataFrame
import pandas as pd

df = pd.DataFrame({
    'Name': ['Joan', 'Matt', 'Jeff', 'Melissa', 'Devi'],
    'Gender': ['Female', 'Male', 'Male', 'Female', 'Female'],
    'House Type': ['Apartment', 'Detached', 'Apartment', None, 'Semi-Detached']
    })

print(df)

print(pd.get_dummies(df['Gender']))

print(pd.get_dummies(df['Name']))

ohe = pd.get_dummies(data=df, columns=['Gender'])
print(ohe)

