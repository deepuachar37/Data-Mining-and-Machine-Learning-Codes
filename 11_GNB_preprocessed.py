
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.read_csv("preprocessed.csv")
winedata = pd.read_csv("preprocessed.csv")

"""
Divinding the data into lables & attributes + test & training data
All the columns of the "preprocessed.csv" are being stored in the X variable except the "label" column,
which is stored in the Y variable. 

X variable: Stores all the attributes
Y variable: Store the label attribute

"""
feature_names = ['fixed acidity', 'volatile acidity',
       'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
       'quality']
X = winedata[feature_names]
y = winedata.label

#Decising the training data & test data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Building GNB Model

from sklearn.naive_bayes import GaussianNB
model = GaussianNB().fit(X_train, y_train)  
print(model)
print('Accuracy of GNB classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'.format(model.score(X_test, y_test))) 

#Prodicting the model
predicted = model.predict(X_test)

#Printing results - Accuracy, Preicison, Recall
from sklearn.metrics import classification_report
print(classification_report(y_test, predicted))

"""
Visializing the Confusion Matrix:
    Step I: Calculate the confusion matrix.
    Step II: Print the confusion matrix.
    Step III: Visualize the confusion matrix.
"""     

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted) 
print(cm)  

#Visualizing the Confusion Matrix

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('Red Wine or White White Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()

print('Code successful')

