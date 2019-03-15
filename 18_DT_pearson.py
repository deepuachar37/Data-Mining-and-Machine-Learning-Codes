
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8


import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.cross_validation import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report





# Function importing Dataset 
def importdata(): 
    winedata = pd.read_csv(r'C:/Users/deepu/Desktop/Extracted/project/preprocessed_pearsons.csv',encoding='utf-8')
    
    # Printing the dataswet shape 
    print ("Dataset Lenght: ", len(winedata)) 
    print ("Dataset Shape: ", winedata.shape) 
      
    # Printing the dataset obseravtions 
    print ("Dataset: ",winedata.head()) 
    return winedata 

def splitdataset(winedata): 
  
    # Seperating the target variable 
    X = winedata.values[:, 1:7] 
    Y = winedata.values[:, 0] 
  
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state = 100) 
      
    return X, Y, X_train, X_test, y_train, y_test 
    
# Function to perform training with entropy. 
def train_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 


# Function to make predictions 
def prediction(X_test, clf_object): 
  
    
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred


# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred))


# Driver code 
def main(): 
      
    # Building Phase 
    data = importdata() 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    clf_entropy = train_using_entropy(X_train, X_test, y_train) 
   
      
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 



# Calling main function 
if __name__=="__main__": 
    main() 






