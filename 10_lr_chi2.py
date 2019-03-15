
# coding: utf-8

# In[2]:


# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 13:19:08 2018

@author: hp
"""
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
wine = pd.read_csv(r'C:/Users/deepu/Desktop/Extracted/project/preprocessed_chi2.csv',encoding='utf-8')
#creating test and train
columns = ['volatile acidity','residual sugar','chlorides','total sulfur dioxide','density','pH','sulphates'] # Declare the columns names
X = wine[columns]
y = wine['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#apply scalling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#fitting logistic regression model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
wine['label'].value_counts()
sns.countplot(x='label',data=wine,palette='hls')
plt.show()
plt.savefig('count_plot')

