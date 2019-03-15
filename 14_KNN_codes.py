
# coding: utf-8

# In[16]:


#Normalised_Knn
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


path='C:/Users/deepu/Desktop/Extracted/project/normalized_merged.csv'
 
df = pd.read_csv(path)

X = df[['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality']]
Y = df[['label']]
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
knn = KNeighborsClassifier(n_neighbors= 3)
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
confusion_knn = confusion_matrix(y_test, y_pred_class)
print(confusion_knn)

accuracy = accuracy_score(y_test, y_pred_class)
print("Accuracy: %.2f%%" % (accuracy *100.0))
print(classification_report(y_test, y_pred_class))



# In[18]:


#chi2_Knn
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


path='C:/Users/deepu/Desktop/Extracted/project/preprocessed_chi2.csv'
 
df1 = pd.read_csv(path)

X = df1[['volatile acidity','residual sugar','chlorides','total sulfur dioxide','density','pH','sulphates']]
Y = df1[['label']]
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
knn = KNeighborsClassifier(n_neighbors= 3)
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
confusion_knn = confusion_matrix(y_test, y_pred_class)
print(confusion_knn)

accuracy = accuracy_score(y_test, y_pred_class)
print("Accuracy: %.2f%%" % (accuracy *100.0))
print(classification_report(y_test, y_pred_class))


# In[12]:


#pearsons_knn
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


path='C:/Users/deepu/Desktop/Extracted/project/preprocessed_pearsons.csv'
 
df2 = pd.read_csv(path)

X = df2[['fixed acidity','volatile acidity','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','sulphates']]
Y = df2[['label']]
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
knn = KNeighborsClassifier(n_neighbors= 3)
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
confusion_knn = confusion_matrix(y_test, y_pred_class)
print(confusion_knn)

accuracy = accuracy_score(y_test, y_pred_class)
print("Accuracy: %.2f%%" % (accuracy *100.0))
print(classification_report(y_test, y_pred_class))


# In[ ]:




