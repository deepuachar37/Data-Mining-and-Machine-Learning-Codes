
# coding: utf-8

# In[2]:


import pandas as pd

merged_df_path = r"C:/Users/deepu/Desktop/Extracted/project/preprocessed.csv"

merged_df = pd.read_csv(merged_df_path)

merged_df[merged_df.columns[0:]].apply(lambda x: x.corr(merged_df['label']))

