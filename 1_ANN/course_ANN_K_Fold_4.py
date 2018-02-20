
# coding: utf-8

# ## cross validation

# In[1]:

def show(id, w=6):
    from IPython.display import Image
    return Image('./pics/class1/{}.jpg'.format(id), width=w*100)


# In[2]:

show(14)


# In[3]:

from IPython.display import HTML
HTML('<img src="./pics/class1/gif1.gif">')


# In[4]:

import pandas as pd
import tensorflow as tf
import theano
import keras 

import numpy as np
import matplotlib.pyplot as plt
from time import time


# ### data preprocessing

# In[5]:

df = pd.read_csv('Artificial_Neural_Networks/Churn_Modelling.csv')
print df.shape
X = df.iloc[:,3:-1].values
y = df.iloc[:,-1].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
print X.shape

# remove one dummy variable to avoid dummy variable trap
print X.shape
X = X[:, 1:]
print X.shape


# ### train test split & feature scaling 

# In[6]:

S = lambda *x: [i.shape for i in x]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print S(X_train,X_test, y_train, y_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# fit on training set
X_train = sc.fit_transform(X_train)
# only transform on test set
X_test = sc.transform(X_test)
print S(X_train,X_test, y_train, y_test)


# ### build k-fold func ANN

# In[7]:

from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
# k-fold cross validation
from sklearn.model_selection import cross_val_score


# In[8]:

def build_classifier():
    classifier = Sequential()
    # first hidden layer
    # classifier.add(Dense(units = 6, input_shape=(11,),  kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units = 6, input_dim=11,  kernel_initializer='uniform', activation='relu'))
    # second hidden layer
    classifier.add(Dense(units = 6,  kernel_initializer='uniform', activation='relu'))
    # ouput layer
    classifier.add(Dense(units = 1,  kernel_initializer='uniform', activation='sigmoid'))
    # compiling the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])
#     print classifier.summary()
    return classifier


# In[9]:

classifier = KerasClassifier(build_fn= build_classifier, batch_size=10, epochs=100)


# In[10]:

if __name__ == "__main__":
    accuracies = cross_val_score(estimator = classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# ### Predicting a single new observation
