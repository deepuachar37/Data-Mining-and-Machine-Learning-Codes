
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  

merged_df_path = r"C:/Users/deepu/Desktop/Extracted/project/preprocessed_chi2.csv"

merged_df = pd.read_csv(merged_df_path)

"""
Divinding the data into lables & attributes + test & training data
All the columns of the dataframe are being stored in the X variable except the "label" column, which is the label column. The drop() method drops this column.
X variable: Stores all the attributes
Y variable: Stores the label attribute

"""

#Deciding the label
X = merged_df.drop('label', axis=1)  
y = merged_df['label']  

#Decising the training data & test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

#Building SVM Model
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train) 


#Prodicting the model
y_pred = svclassifier.predict(X_test)  

"""
Visializing the Confusion Matrix:
    Step I: Calculate the confusion matrix.
    Step II: Print the confusion matrix.
    Step III: Visualize the confusion matrix.
"""      

#Confusion Matrix
cm = confusion_matrix(y_test,y_pred)

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

#Printing results - Accuracy, Preicison, Recall
print(classification_report(y_test,y_pred)) 

print('Code successful')

