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
warnings.filterwarnings("ignore")


railfall = pd.read_csv('Data.csv')
affect = pd.read_csv('Sheet_1_data.csv')

MasterData = pd.merge(railfall,affect,on= 'Year',how='right').fillna(0)
print (list(MasterData))
Y = MasterData['Total deaths']
X = MasterData[['Year', 'occurrence', 'Total deaths', 'Injured', 'Affected', 'Homeless', 'Total affected', "Total damage  ('000 US$)", 'Rainfall']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
lg = LogisticRegression().fit(X_train,y_train)
gnb = GaussianNB().fit(X_train,y_train)
dt =  tree.DecisionTreeRegressor().fit(X_train,y_train)
svm = clf = SVC(gamma='auto').fit(X_train,y_train)
neigh = KNeighborsClassifier(n_neighbors=3).fit(X_train,y_train)

pickle.dump(lg, open('Logestic.p', 'wb'))
pickle.dump(gnb, open('Baysean.p', 'wb'))
pickle.dump(dt, open('decisionTree.p', 'wb'))
pickle.dump(svm, open('svm.p', 'wb'))
pickle.dump(neigh, open('knn.p', 'wb'))
results = {}
print('Accuracy of Diffrent Algorithms')
score = lg.score(X_test, y_test)
print("%s : %f %%" % ('LR', score*100))
results['Logistic Regression'] = 100*score

score = gnb.score(X_test, y_test)
print("%s : %f %%" % ('GNB', score*100))
results['GaussianNB'] = 100*score

score = dt.score(X_test, y_test)
print("%s : %f %%" % ('DT', score*100))
results['Decision Tree'] = 100*score

score = svm.score(X_test, y_test)
print("%s : %f %%" % ('svm', score*100))
results['Support Vector Machine'] = 100*score

score = neigh.score(X_test, y_test)
print("%s : %f %%" % ('KNN', score*100))
results['KNN'] = 100*score

F1Score={}
MSE={}
res = lg.predict(X_test)
FS=f1_score(y_test, res, average='macro')
print ('F1 Score of Logestic Regression is   : %f %%'% (FS*100))
F1Score['Logistic Regression'] = 100*FS
MSE['Logistic Regression'] = mean_squared_error(y_test, res)

res = gnb.predict(X_test)
FS=f1_score(y_test, res, average='macro')
print ('F1 Score of GaussianNB is   : %f %%'% (FS*100))
F1Score['GaussianNB'] = 100*FS
MSE['GaussianNB'] = mean_squared_error(y_test, res)

res = dt.predict(X_test)
FS=f1_score(y_test, res, average='macro')
print ('F1 Score of Decision Tree is   : %f %%'% (FS*100))
F1Score['Decision Tree'] = 100*FS
MSE['Decision Tree'] = mean_squared_error(y_test, res)

res = svm.predict(X_test)
FS=f1_score(y_test, res, average='macro')
print ('F1 Score of Support Vector Machine  is   : %f %%'% (FS*100))
F1Score['Support Vector Machine'] = 100*FS
MSE['Support Vector Machinee'] = mean_squared_error(y_test, res)

res = neigh.predict(X_test)
FS=f1_score(y_test, res, average='macro')
print ('F1 Score of KNN  is   : %f %%'% (FS*100))
F1Score['KNN'] = 100*FS
MSE['KNN'] = mean_squared_error(y_test, res)

D=results
plt.figure();
plt.bar(range(len(D)), list(D.values()), align='center',color= 'r')
plt.xticks(range(len(D)), list(D.keys()))
plt.title('Accuracy of diffrent algorithms(tested for random 30% data)')
plt.show()


D= F1Score
plt.figure();
plt.bar(range(len(D)), list(D.values()), align='center',color= 'g')
plt.xticks(range(len(D)), list(D.keys()))
plt.title('F1 Score of diffrent algorithms(tested for random 30% data)')
plt.show()

D= MSE
plt.figure();
plt.bar(range(len(D)), list(D.values()), align='center',color= 'k')
plt.xticks(range(len(D)), list(D.keys()))
plt.title('Mean Squared error of diffrent algorithms(tested for random 30% data)')
plt.show()

plt.plot(MasterData['Year'],MasterData['Total deaths'],MasterData['Year'],MasterData['Injured'],MasterData['Year'],MasterData['Affected'])
plt.legend(('Total Deaths)', 'Injured', 'Affected'),loc='upper right')
plt.title('Damage year-wise')
plt.show()
plt.fill_between(MasterData['Year'],MasterData['Homeless'],color='orange')
plt.title('Year-Wise Homeless due to floods')
plt.show()


plt.bar(MasterData['Year'],MasterData['Rainfall'], align='center', color = 'r')
plt.title('Rainfall in Year')
plt.show()

plt.bar(MasterData['Year'],MasterData['Total affected'], align='center', color = 'g')
plt.title('Total Affected by flood')
plt.show()

plt.bar(MasterData['Year'],MasterData["Total damage  ('000 US$)"], align='center', color = 'b')
plt.title('Total Loss in USD')
plt.show()
