
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


winedata = pd.read_csv(r'C:/Users/deepu/Desktop/Extracted/project/preprocessed.csv',encoding='utf-8')

p = winedata.drop('label', axis=1)
q = winedata['label']

#http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
# Feature extraction
test = SelectKBest(score_func=chi2, k=7)
fit = test.fit(p, q)


# Summarize scores
np.set_printoptions(precision=4)
print(fit.scores_)


# Feature extraction
model = LogisticRegression()
rfe = RFE(model, 7)
fit = rfe.fit(p, q)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

t=p.drop(p.columns[[0,2,5,10,11]],axis=1)

t.head()

u=winedata.drop(winedata.columns[[1,2,3,4,5,6,7,8,9,10,11,12]],axis=1)

u.head()

v = pd.concat([u, t], axis=1)

print(v)

v.to_csv('feature_selection2.csv', encoding='utf-8', index=False)






