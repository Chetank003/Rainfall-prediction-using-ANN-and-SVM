import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,mean_squared_error
import warnings

railfall = pd.read_csv('Data.csv')
affect = pd.read_csv('Sheet_1_data.csv')

MasterData = pd.merge(railfall,affect,on= 'Year',how='right').fillna(0)
print (list(MasterData))
Y = MasterData['Total deaths']
X = MasterData[['Year', 'occurrence', 'Total deaths', 'Injured', 'Affected', 'Homeless', 'Total affected', "Total damage  ('000 US$)", 'Rainfall']]


svm = clf = SVC(gamma='auto').fit(X,Y)


