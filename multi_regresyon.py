# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:32:22 2022

@author: emrey
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('winequality-red.csv')


#regresyon uygulanacak verileri ayırma
x = veriler.iloc[:,0:-1] #bağımsız değişken
y = veriler.iloc[:,11:12] #bağımlı değişken


#verileri numpy dizisine dönüştürme
X = x.values
Y = y.values

### Verilerin Egitim ve Test İcin Bölünmesi ###

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

### Çoklu Doğrusal Regresyon Uygulanması ###
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

### Tahmin değerlerinin yazdırılması ###
y_pred = regressor.predict(x_test)
print(y_pred)



### Backward Elimination Yöntemi ile İyileştirme ###
import statsmodels.api as sm 
X = np.append(arr = np.ones((1599,1)).astype(int), values=x, axis=1 )

X_l = x.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(Y,X_l).fit()
print(model.summary())

X_l = x.iloc[:,[1,2,3,4,5,6,7,8,9,10]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(Y,X_l).fit()
print(model.summary())

X_l = x.iloc[:,[1,2,4,5,6,7,8,9,10]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(Y,X_l).fit()
print(model.summary())

X_l = x.iloc[:,[1,4,5,6,7,8,9,10]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(Y,X_l).fit()
print(model.summary())

#Eliminasyon Sonrası İyileşme Ölçmek için test ve train verilerinin güncellenmesi

x_train2 = x_train.iloc[:,4:11]
x_train3 = x_train.iloc[:,1:2]
x_train4 = pd.concat([x_train3,x_train2], axis=1)

x_test2 = x_test.iloc[:,4:11]
x_test3 = x_test.iloc[:,1:2]
x_test4 = pd.concat([x_test3,x_test2], axis=1)

# Tahmin değerinin yazdırılması
regressor.fit(x_train4,y_train)

y_pred2 = regressor.predict(x_test4)


