# Importing Libraries
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve

# Create a dataframe with the data from Kaggle .csv file.
df = pd.read_csv((r"/Volumes/Backup_Volume/GitHub Repository/GitHub Python Projects/Python-Projects-Repository/PhD ML 2nd Paper [Sample Code]/PeX_Dataset [Preprocessing]/Feature Selection/KaggleV2-May-2016.csv"), encoding="ISO-8859-1", low_memory=False)
df.head()
# print(df)

# Investigate the No-Show column with value_counts
print(df['No-show'].value_counts())
df['OUTPUT_LABEL'] = " "

# Define a binary column OUTPUT_LABEL to indicate Yes = 1, No = 0
df['OUTPUT_LABEL'] = (df['No-show'] == 'Yes').astype('int')

# Function to check the prevalence of our OUTPUT_LABEL column of the df dataframe
def calc_prevalence(y):
 return (sum(y)/len(y))

# Investigate the OUTPUT_LABEL column with value_counts. The statement that follows will return the total 
# values representing each 'OUTPUT_LABEL'.
print(df['OUTPUT_LABEL'].value_counts())

# Investigate the prevalence of our OUTPUT_LABEL column of the df dataframe
calc_prevalence(df['OUTPUT_LABEL'].values)
print(calc_prevalence(df['OUTPUT_LABEL'].values))

# Manipulate the datetime columns by looking at the first 5 rows of ScheduledDay and AppointmentDay
print(df['ScheduledDay'].head())
print(df['AppointmentDay'].head())

df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], format = '%Y-%m-%dT%H:%M:%SZ', errors = 'coerce')
print(df['ScheduledDay'].head())
assert df['ScheduledDay'].isnull().sum() == 0, 'missing ScheduledDay dates'

df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], format = '%Y-%m-%dT%H:%M:%SZ', errors = 'coerce')
print(df['AppointmentDay'].head())
assert df['AppointmentDay'].isnull().sum() == 0, 'missing AppointmentDay dates'

# Currently there are ~40k appointments that were scheduled after the appointment datetime
print((df['ScheduledDay'] > df['AppointmentDay']).sum())

# Shift all the appointment times to the end of the day.
df['AppointmentDay'] = df['AppointmentDay'] +pd.Timedelta('1d') - pd.Timedelta('1s')
print(df['AppointmentDay'])

# Basically you can break apart the date and get the year, month, week of year, day of month, 
# hour, minute, second, etc. You can also get the day of the week (Monday = 0, Sunday = 6). 
# Note be careful with week of year because the first few days of the year may be 53 if that week 
# begins in the prior year. Let’s apply some of these properties to both of our datetime columns.

df['ScheduledDay_year'] = df['ScheduledDay'].dt.year
df['ScheduledDay_month'] = df['ScheduledDay'].dt.month
df['ScheduledDay_week'] = df['ScheduledDay'].dt.week
df['ScheduledDay_day'] = df['ScheduledDay'].dt.day
df['ScheduledDay_hour'] = df['ScheduledDay'].dt.hour
df['ScheduledDay_minute'] = df['ScheduledDay'].dt.minute
df['ScheduledDay_day_of_week'] = df['ScheduledDay'].dt.day_of_week

df['AppointmentDay_year'] = df['AppointmentDay'].dt.year
df['AppointmentDay_month'] = df['AppointmentDay'].dt.month
df['AppointmentDay_week'] = df['AppointmentDay'].dt.week
df['AppointmentDay_day'] = df['AppointmentDay'].dt.day
df['AppointmentDay_hour'] = df['AppointmentDay'].dt.hour
df['AppointmentDay_minute'] = df['AppointmentDay'].dt.minute
df['AppointmentDay_day_of_week'] = df['AppointmentDay'].dt.day_of_week

# Verification code to show that the previous statements work.
print(df[['ScheduledDay', 'ScheduledDay_year', 'ScheduledDay_month', 'ScheduledDay_week', 
'ScheduledDay_day', 'ScheduledDay_hour', 'ScheduledDay_minute', 'ScheduledDay_day_of_week']].head())

print(df[['AppointmentDay', 'AppointmentDay_year', 'AppointmentDay_month', 'AppointmentDay_week', 
'AppointmentDay_day', 'AppointmentDay_hour', 'AppointmentDay_minute', 'AppointmentDay_day_of_week']].head())

# # At this point it would be good to explore our dates a bit.
# print(df.groupby('ScheduledDay_year').size())
# print(df.groupby('ScheduledDay_month').size())
# print(df.groupby('ScheduledDay_week').size())
# print(df.groupby('ScheduledDay_day').size())
# print(df.groupby('ScheduledDay_hour').size())
# print(df.groupby('ScheduledDay_minute').size())
# print(df.groupby('ScheduledDay_day_of_week').size())

# print(df.groupby('AppointmentDay_year').size())
# print(df.groupby('AppointmentDay_month').size())
# print(df.groupby('AppointmentDay_week').size())
# print(df.groupby('AppointmentDay_day').size())
# print(df.groupby('AppointmentDay_hour').size())
# print(df.groupby('AppointmentDay_minute').size())
# print(df.groupby('AppointmentDay_day_of_week').size())

# Let’s quickly check if dayofweek is predictive of no-show:
print(df.groupby('AppointmentDay_day_of_week').apply(lambda g: calc_prevalence(g.OUTPUT_LABEL.values)))
# print(df.loc[df['OUTPUT_LABEL'] == 0])

# Another nice thing with pandas datetime representation is that you can calculate the ‘time’ between datetimes. 
# Let’s create a new feature that is the number of days between the scheduled date and the appointment date.
df['delta_days'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.total_seconds()/(60*60*24)
# print(df)

# plotting two histograms on the same axis
df2 = df.loc[df['OUTPUT_LABEL'] == 0]
print(df2)
x = df2['delta_days']
df3 = df.loc[df['OUTPUT_LABEL'] == 1]
print(df3)
y = df3['delta_days']

# bins = np.linspace(10, 100)
bins = range(0,60,1)
# print(bins)

plt.hist(x, bins, alpha=0.5, label='Missed', density=True)
plt.hist(y, bins, alpha=0.5, label='Not Missed', density=True)
plt.legend()
plt.xlim(0,40)
plt.show()

# We are now ready to split our samples and train a model!
# ####################################################################################
# Split Samples
# ####################################################################################
# For simplicity, I’ll just split into two datasets: train (70%) and validation (30%). 
# It is important to shuffle your samples because you may have been given the data in 
# order of dates.
# ####################################################################################

# shuffle the samples
# df = df.sample(n = len(df), random_state = 42)
# df = df.reset_index(drop = True)
# df_valid = df.sample(frac = 0.3, random_state = 42)
# # print(df_valid)
# df_train = df.drop(df_valid.index)
# # print(df_train)

# We can check the prevalence is about 20% in each:
# print('Valid prevalence(n = %d):%.3f'%(len(df_valid), calc_prevalence(df_valid['OUTPUT_LABEL'].values)))
# print('Train prevalence(n = %d):%.3f'%(len(df_train), calc_prevalence(df_train['OUTPUT_LABEL'].values)))

# # Given this data comes from just Apr-Jun 2016 and their are no appointment times, we will just use these columns:
# col2use = ['ScheduledDay_day', 'ScheduledDay_hour', 'ScheduledDay_minute', 'ScheduledDay_day_of_week', 
# 'AppointmentDay_day','AppointmentDay_day_of_week', 'delta_days']

# # We can now build our X (inputs) and Y(output) for training and validation:
# X_train = df_train[col2use].values
# X_valid = df_valid[col2use].values
# y_train = df_train['OUTPUT_LABEL'].values
# y_valid = df_valid['OUTPUT_LABEL'].values
# print('Training shapes:',X_train.shape, y_train.shape)
# print('Validation shapes:',X_valid.shape, y_valid.shape)

# # Train a Machine Learning Model
# # #################################### Pay Attention for Future Work!!! ####################################
# # Since the focus of this post is about the datetime features, we will just train a random forest model here. 
# # Note if you want to use other types of models, you may need to scale or normalize your data. Another thing 
# # you may want to do is convert the dayofweek into a categorical variable via one-hot encoding. We don’t need 
# # to do these things for a tree-based method though.

# rf = RandomForestClassifier(max_depth = 5, n_estimators=100, random_state = 42)
# rf.fit(X_train, y_train)

# # We can then get our predictions with:
# y_train_preds = rf.predict_proba(X_train)[:,1]
# print(y_train_preds)
# y_valid_preds = rf.predict_proba(X_valid)[:,1]
# print(y_valid_preds)

# # Here we will evaluate performance of the model. For that reason we created the two functions 
# # calc_specificity() and print_report().
# def calc_specificity(y_actual, y_pred, thresh):
#     # calculates specificity
#     return sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)

# def print_report(y_actual, y_pred, thresh):
#     auc = roc_auc_score(y_actual, y_pred)
#     accuracy = accuracy_score(y_actual, (y_pred > thresh))
#     recall = recall_score(y_actual, (y_pred > thresh))
#     precision = precision_score(y_actual, (y_pred > thresh))
#     specificity = calc_specificity(y_actual, y_pred, thresh)
#     print('AUC:%.3f'%auc)
#     print('accuracy:%.3f'%accuracy)
#     print('recall:%.3f'%recall)
#     print('precision:%.3f'%precision)
#     print('specificity:%.3f'%specificity)
#     print('prevalence:%.3f'%calc_prevalence(y_actual))
#     print(' ')
#     return auc, accuracy, recall, precision, specificity

# # We set the threshold to be the same as the prevalence.
# thresh = 0.201

# print(' ')
# print('Random Forest')
# print('Training:')
# print_report(y_train, y_train_preds, thresh)
# print('Validation:')
# print_report(y_valid, y_valid_preds, thresh)

