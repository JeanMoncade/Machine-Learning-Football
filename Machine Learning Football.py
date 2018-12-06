# -*- coding: utf-8 -*
""" Created on Wed Aug 15 17:13:15 2018

@author: jeanm 
"""

'''Import Libraries import numpy as np import pandas as pd'''
import numpy as np 
import pandas as pd

'''import Dataset dataset = pd.read_csv("players.data1.csv")'''
dataset = pd.read_csv("players.data1.csv")
X = dataset.iloc[:,[0, 2, 3, 4]] 
y = dataset.iloc[:,1]

'''Gestion des valeurs manquantes'''
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0) 
imputer.fit(X.iloc[:, 5:6])
X.iloc[:, 5:6] = imputer.transform(X.iloc[:, 5:6])

'''Arrondir les nouvelles valeurs ajoutées'''
 X.iloc[:,5]= X.iloc[:,5].round()
 
'''
#Visualisation exploratoire
sns.lmplot(x = "Buts par 90 min", y = "Valeur marchande", data = dataset, logistic = False) sns.lmplot(x = "Âge", y = "Valeur marchande", data = dataset, logistic = False) 
sns.lmplot(x = "Passes décisives par 90 min", y = "Valeur marchande", data = dataset, logistic = False) 
sns.lmplot(x = "Taux de conversion but/tir", y = "Valeur marchande", data = dataset, logistic = False) 
sns.lmplot(x = "Contrat expiration", y = "Valeur marchande", data = dataset, logistic = False) 
sns.lmplot(x = "Buts par 90 min", y = "Âge", data = dataset, logistic = False) 
sns.lmplot(x = "Taux de conversion but/tir", y = "Âge", data = dataset, logistic = False)
'''

'''Séparer dataset en training set et test set'''
from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

'''Appliquer notre modèle''' 
from sklearn.linear_model import LinearRegression 
regressor_lr = LinearRegression() 
regressor_lr.fit(X_train,y_train)
y_pred_lr = regressor_lr.predict(X_test)
overview_y_pred = y_pred_lr 
overview_y_test = y_test
overview = pd.DataFrame(data=np.column_stack((overview_y_pred,overview_y_test)), 
                        columns=["Pred", "Valeurs réelles"]) 
overview.head()

'''Scoring Mean Square ERROR''' 
ecart = ((y_pred_lr - y_test)**2)**(1/2) 
ecart.mean()

'''Ajout d'une constante'''
 X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
 
'''Modèle optimun'''
import statsmodels.formula.api as sm 
regressor_OLS = sm.OLS(endog = y, exog = X).fit() 
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 4]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 
regressor_OLS.summary() 

# meilleure modèle ci_dessus
X_opt = X[:, [0, 2]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 
regressor_OLS.summary()