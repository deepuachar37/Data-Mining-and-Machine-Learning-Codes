
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn import preprocessing

#Importing files
red_wine_path = r"C:/Users/deepu/Desktop/Extracted/project/winequality-red_raw.csv"
white_wine_path  = r"C:/Users/deepu/Desktop/Extracted/project/winequality-white_raw.csv"

#Reading files
white_wine_df = pd.read_csv(white_wine_path)
red_wine_df = pd.read_csv(red_wine_path)

print(list(white_wine_df))
print(list(red_wine_df))

#Random sampling of white wine file
sampled_white_df = white_wine_df.sample(1599, replace = False)
print(list(sampled_white_df))

#Data integration - row-wise
merged_df = pd.concat([sampled_white_df,red_wine_df])

#Replacing string labels to string 1 & 0
merged_df.loc[merged_df['label'].str.contains('White', na = False),'label'] = '1'
merged_df.loc[merged_df['label'].str.contains('Red', na = False),'label'] = '0'

#Data type of the replaced labels
merged_df.dtypes

#Converting the string labels into numeric labels
merged_df = merged_df.convert_objects(convert_numeric=True)

merged_df.dtypes


list(merged_df)    

#Export the preprocessed file as csv   
merged_df.to_csv('merged.csv')

#Count no. of NaNs
na_count = merged_df.isnull().sum()
print(na_count)

#Normalizing data between 0 & 1
for column in merged_df.columns[1:]:
    x = merged_df[column].values
    x = x.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    merged_df[column] = x_scaled
    merged_df = merged_df
    
merged_df.to_csv('normalized_merged.csv')

#Binning basis the min & max
#for column in merged_df.columns[1:]:
     #binwidth = int((max(merged_df[column]) - min(merged_df[column]))/5.0)
     #bins = range(min(merged_df[column]),max(merged_df[column]), binwidth)
     #group_names = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
     #merged_df[column] = pd.cut(merged_df[column], bins, labels = group_names)

#Correlation between variables (for ranking)
#Correlation with label for Feature Selection
#merged_df[merged_df.columns[0:]].corr()['label'][:-1]

#Choosing columns after Feature Selection

#Know final rows and columns
merged_df.shape


#merged_df.to_csv('preprocessed.csv')

#Generating thecorrelation matrix
#merged_df[merged_df.columns[0:]].apply(lambda x: x.corr(merged_df['label']))

#merged_df.drop()

#merged_df.to_csv('preprocessed_pearsons.csv')

