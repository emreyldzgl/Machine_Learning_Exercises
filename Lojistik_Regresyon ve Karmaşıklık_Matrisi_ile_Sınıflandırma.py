# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 18:08:38 2022

@author: emrey
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veri yukleme
veriler = pd.read_csv('wheat-seeds.csv')


print(veriler)

x = veriler.iloc[:,0:7].values #bağımsız değişkenler
y = veriler.iloc[:,7:].values #bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

## Lojistik Regresyon ##

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

## Confusion Matrix(Karmaşıklık Matrisi) ##

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)





