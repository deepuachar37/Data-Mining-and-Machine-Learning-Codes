
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np

path1='C:/Users/deepu/Desktop/Extracted/project/normalized_merged.csv'

df=pd.read_csv(path1)
bin1=[0,0.27,0.54,1]
bin2=[0,0.24,0.48,1]
bin3=[0,0.21,0.42,1]
bin4=[0,0.13,0.26,1]
bin5=[0,0.23,0.46,1]
bin6=[0,0.30,0.60,1]
bins=[0,0.33,0.66,1]
group_names=['Average','Mediocre','Very Good']
df['fixed_acidity']=pd.cut(df['fixed_acidity'],bins,labels=group_names)
df['volatile_acidity']=pd.cut(df['volatile_acidity'],bin1,labels=group_names)
df['citric_acid']=pd.cut(df['citric_acid'],bin2,labels=group_names)
df['residual_sugar']=pd.cut(df['residual_sugar'],bin3,labels=group_names)
df['chlorides']=pd.cut(df['chlorides'],bin1,labels=group_names)
df['free_sulfur_dioxide']=pd.cut(df['free_sulfur_dioxide'],bin4,labels=group_names)
df['total_sulfur_dioxide']=pd.cut(df['total_sulfur_dioxide'],bin5,labels=group_names)
df['density']=pd.cut(df['density'],bins,labels=group_names)
df['pH']=pd.cut(df['pH'],bins,labels=group_names)
df['sulphates']=pd.cut(df['quality'],bins,labels=group_names)
df['alcohol']=pd.cut(df['alcohol'],bin6,labels=group_names)
df['quality']=pd.cut(df['quality'],bins,labels=group_names)


# In[6]:


df


# In[8]:


df.to_csv('./des.csv')

