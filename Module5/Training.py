import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVR
from joblib import dump,load

df=pd.read_csv('data1modify.csv')
df=df.dropna()
a=list(df.columns)
X=df[[a[0],a[1],a[2],a[3],a[4],a[5]]]
Y = df[a[6]]

svm = clf = SVR(gamma='auto').fit(X,Y)
dump(svm,'svm.model')

