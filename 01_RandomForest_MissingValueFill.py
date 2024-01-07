import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import warnings # Ignore warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # For displaying Chinese labels
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(4)

df = pd.read_csv(r'first.csv')
df = df.drop_duplicates(subset = df.columns.tolist()[1:-1], keep = 'first')
df = df.dropna(thresh=0.4*df.shape[0], axis=1)


def convert_to_int(value):
    try:
        return int(''.join(filter(str.isdigit, value)))
    except ValueError:
        return 0

df['VISCODE'] = df['VISCODE'].apply(convert_to_int)

# Quantify the 'PTGENDER' column
conditions = [df['PTGENDER'] == 'Male',
              df['PTGENDER'] == 'Female']
choices = [1, 2]
df['PTGENDER'] = np.select(conditions, choices, default=df['PTGENDER'])

# Quantify the 'PTETHCAT' column
conditions = [df['PTETHCAT'] == 'Unknown',
              df['PTETHCAT'] == 'Hisp/Latino',
              df['PTETHCAT'] == 'Not Hisp/Latino']
choices = [0, 1, 2]
df['PTETHCAT'] = np.select(conditions, choices, default=df['PTETHCAT'])

# Quantify the 'PTRACCAT' column
conditions = [df['PTRACCAT'] == 'Unknown',
              df['PTRACCAT'] == 'White',
              df['PTRACCAT'] == 'Black',
              df['PTRACCAT'] == 'Asian',
              df['PTRACCAT'] == 'Am Indian/Alaskan',
              df['PTRACCAT'] == 'Hawaiian/Other PI',
              df['PTRACCAT'] == 'More than one']
choices = [0, 1, 2,3,4,5,6]
df['PTRACCAT'] = np.select(conditions, choices, default=df['PTRACCAT'])


# Quantify the 'PTMARRY' column
conditions = [df['PTMARRY'] == 'Unknown',
              df['PTMARRY'] == 'Married',
              df['PTMARRY'] == 'Divorced',
              df['PTMARRY'] == 'Widowed',
              df['PTMARRY'] == 'Never married',]
choices = [0, 1, 2,3,4]
df['PTMARRY'] = np.select(conditions, choices, default=df['PTMARRY'])

# Quantify the 'DX' column
conditions = [df['DX'] == 'CN',
              df['DX'] == 'Dementia',
              df['DX'] == 'MCI',]
choices = [1, 2,3]
df['DX'] = np.select(conditions, choices, default=df['DX'])


# Quantify the 'DX_bl' column
conditions = [df['DX_bl'] == 'CN',
              df['DX_bl'] == 'AD',
              df['DX_bl'] == 'SMC',
              df['DX_bl'] == 'EMCI',
              df['DX_bl'] == 'LMCI']
choices = [1, 2,3,4,5]
df['DX_bl'] = np.select(conditions, choices, default=df['DX_bl'])

df.to_csv(r'quantified_data.csv',index = False, header=True,encoding ="utf_8_sig",errors='strict')

lianghua= pd.read_csv(r'quantified_data.csv')
lianghua.isna().sum()  # Count the number of missing values
sindex = np.argsort(lianghua.isna().sum().values.tolist()) # Sort columns with missing values from least to most
object_data=lianghua.select_dtypes(include=['object'])  # Extract columns that need to be quantified


# import time
# Record the start time
# start_time = time.time()

for i in sindex: # Sort by the number of missing values, iterating from small to large
    if lianghua.iloc[:,i].isna().sum() == 0: # Filter out rows with no missing values
             continue # Skip the current iteration directly
    df = lianghua # Copy the data from lianghua
    fillc = df.iloc[:,i] # Extract the i-th column to be used as the y variable
    df = df.iloc[:,df.columns != df.columns[i]] # Data excluding this column, to be used as X
    df_0 = SimpleImputer(missing_values=np.nan, strategy="constant",fill_value=0).fit_transform(df)
    ytrain = fillc[fillc.notnull()] # In the fillc column, non-NAN values are used as Y_train
    ytest = fillc[fillc.isnull()] # In the fillc column, NAN values are used as Y_test
    xtrain=df_0[ytrain.index,:] # In df_0 (already filled with 0), rows where fillc column is not NAN are used as X_train
    xtest=df_0[ytest.index,:] # In df_0 (already filled with 0), rows where fillc is equal to NAN are used as X_test

    rfc = RandomForestRegressor()
    rfc.fit(xtrain,ytrain)
    ypredict = rfc.predict(xtest) # Ytest is determined based on Xtest, resulting in Ypredict

    lianghua.loc[lianghua.iloc[:,i].isnull(),lianghua.columns[i]] = ypredict
# Replace the rows in the data_copy where the i-th column is empty with Ypredict
# lianghua.isna().sum() # Count the number of missing values
# end_time = time.time()
# Calculate the time difference
# run_time = end_time - start_time
# Print the runtime
# print("Program run time: ", run_time, " seconds")


lianghua['DX'] = lianghua['DX'].round().astype(int)
save_path = r'filled_data_rf.csv'  # Save path and filename
lianghua.to_csv(save_path, index=False)