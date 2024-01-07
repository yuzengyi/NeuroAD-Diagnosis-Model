import pandas as pd
import numpy as np
# Read data from 'filled_data_rf.csv' file
df= pd.read_csv(r'filled_data_rf.csv')
ID=df.loc[:,'RID']
DX=df.loc[:,'DX']
DX_bl=df.loc[:,'DX_bl']
VISCODE=df.loc[:,'VISCODE']
data_x = df.drop(['RID','DX','DX_bl','VISCODE'], axis=1)
# Calculate outliers using standard deviation
df1 = np.abs((data_x - data_x.mean())) > (3*data_x.std())

# Initialize lists and DataFrames
yichang = []  # Used to store the number of outliers in each column
pff = pd.DataFrame()  # Used to store variables that can be processed with relatively few outliers
dff = pd.DataFrame()  # Used to store variables that should not be processed due to relatively many outliers
for m in range(df1.shape[1]):  # Calculate the number of outliers in each column
    number = np.sum(df1.iloc[:, m])
    yichang.append(number)
# Identify columns with fewer outliers
t = pd.DataFrame(yichang) < 0.01 * data_x.shape[0]
t1 = t.iloc[:, 0]
# Split columns into those that can be processed and those that can't
for i in range(df1.shape[1]):
    if t1[i] == True:
        pff = pd.concat([pff, data_x.iloc[:, i]], axis=1)  # Put together variables that can be processed
    if t1[i] == False:
        dff = pd.concat([dff, data_x.iloc[:, i]], axis=1)  # Put together variables that cannot be processed

# Function to deal with outliers by replacing them with mode
def deal_abnormal(df):
    wu = np.abs((df - df.mean())) > (3*df.std())
    for i in range(len(df)):
        for j in range(len(df.columns)):
            if wu.iloc[i, j] == True:
                df.iloc[i, [j]] = df.iloc[:, j].mode().values[0] # Fill with the mode
    return df
# Process columns with fewer outliers
pff1 = deal_abnormal(pff)
dx1 = pd.concat([dff,pff1],axis = 1) # Combine data
data4=pd.concat([ID,VISCODE,DX_bl,DX,dx1],axis = 1)
# Function to normalize data
def Norm(df):
    return (df - df.mean()) / df.var()
# Normalize data and concatenate it with other columns
dfnorm = Norm(data4.drop(['RID','DX','DX_bl','VISCODE'], axis=1))
dfnor=pd.concat([ID,VISCODE,dfnorm,DX_bl,DX],axis = 1)
# Save normalized data to 'normalize_data.csv' file
dfnor.to_csv(r'normalize_data.csv',index = False, header=True,encoding
="utf_8_sig",errors='strict')
# Read the normalized data from 'normalize_data.csv'
guiyi= pd.read_csv(r'normalize_data.csv')
# Calculate Spearman correlation coefficients
spearman=guiyi.drop(['RID'], axis=1).corr('spearman')
spear_DX=abs(spearman.loc[:,'DX'])
spear_DX1 = spear_DX.sort_values(ascending=False)
spear_DX1.to_csv(r'Spearman_correlation_coefficient_results.csv',encoding = 'gbk',index =True)

# Import Kruskal-Wallis test
from scipy.stats import kruskal

def kruskal_test(group_data, target_data):
    groups = group_data.unique()
    result = {}

    for group in groups:
        group_values = target_data[group_data == group]
        statistic, p_value = kruskal(group_values, target_data)
        result[group] = {"statistic": statistic, "p-value": p_value}

    return result

# Perform Kruskal-Wallis test for different groups
result_gender = kruskal_test(guiyi['PTGENDER'], guiyi['DX'])
result_ethcat = kruskal_test(guiyi['PTETHCAT'], guiyi['DX'])
result_raccat = kruskal_test(guiyi['PTRACCAT'], guiyi['DX'])
result_marry = kruskal_test(guiyi['PTMARRY'], guiyi['DX'])
import pandas as pd
# Function to save Kruskal-Wallis test results to an Excel file
def save_to_excel(results, sheet_name, writer):
    df = pd.DataFrame(columns=['Group', 'h-value', 'p-value'])

    for group, values in results.items():
        df = df.append({'Group': group, 'h-value': values['statistic'], 'p-value': values['p-value']}, ignore_index=True)

    df.to_excel(writer, sheet_name=sheet_name, index=False)

# Create an ExcelWriter object and specify the file name for saving
writer = pd.ExcelWriter('KW_correlation_coefficient.xlsx', engine='xlsxwriter')

# Save results to different sheets in the Excel file
save_to_excel(result_gender, 'PTGENDER_p_values', writer)
save_to_excel(result_ethcat, 'PTETHCAT_p_values', writer)
save_to_excel(result_raccat, 'PTRACCAT_p_values', writer)
save_to_excel(result_marry, 'PTMARRY_p_values', writer)

# Close the ExcelWriter object
writer.save()