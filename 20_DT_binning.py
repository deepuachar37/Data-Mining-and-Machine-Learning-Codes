
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import graphviz

winedata = pd.read_csv(r'C:/Users/deepu/Desktop/Extracted/project/Final_ET.csv',encoding='utf-8')

winedata.head()

winedata.info()

p = winedata.drop('label', axis=1)
q = winedata['label']

winedata['fixed_acidity'],_ = pd.factorize(winedata['fixed_acidity'])
winedata['volatile_acidity'],_ = pd.factorize(winedata['volatile_acidity'])
winedata['citric_acid'],_ = pd.factorize(winedata['citric_acid'])
winedata['residual_sugar'],_ = pd.factorize(winedata['residual_sugar'])
winedata['chlorides'],_ = pd.factorize(winedata['chlorides'])
winedata['free_sulfur_dioxide'],_ = pd.factorize(winedata['free_sulfur_dioxide'])
winedata['total_sulfur_dioxide'],_ = pd.factorize(winedata['total_sulfur_dioxide'])
winedata['density'],_ = pd.factorize(winedata['density'])
winedata['pH'],_ = pd.factorize(winedata['pH'])
winedata['sulphates'],_ = pd.factorize(winedata['sulphates'])
winedata['alcohol'],_ = pd.factorize(winedata['alcohol'])
winedata['quality'],_ = pd.factorize(winedata['quality'])

winedata.info()

winedata.head()

p = winedata.drop('label', axis=1)
p.head()

feature = ['fixed_acidity', 'volatile_acidity', 'citric_acid',
       'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide','density','pH','sulphates','alcohol','quality']
r = winedata[feature]
#target=['label']
#s = winedata[target]
#print(s)
#t= winedata.label

X_train, X_test, y_train, y_test = train_test_split(r, q, test_size=0.3, random_state=0)

model = DecisionTreeClassifier().fit(X_train, y_train)  
print(model)
print('Accuracy of DT classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
print('Accuracy of DT classifier on test set: {:.2f}'.format(model.score(X_test, y_test))) 

predicted = model.predict(X_test)

print(classification_report(y_test, predicted))

print(confusion_matrix(y_test, predicted))  

feature_names = r.columns

dot_data = tree.export_graphviz(model, out_file=None, filled=True, rounded=True,
                                feature_names=feature_names,  
                                class_names=q)
graph = graphviz.Source(dot_data)  
graph

