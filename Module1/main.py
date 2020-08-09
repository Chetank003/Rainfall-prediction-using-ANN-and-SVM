import pandas as pd
import matplotlib.pyplot as plt

railfall = pd.read_csv('rainfall.csv')
affect = pd.read_csv('affect.csv')
MasterData = pd.merge(railfall,affect,on= 'Year',how='right').dropna()

print (list(MasterData))
Y = MasterData['Lives Lost (in Nos.)']
X = MasterData[['APR', 'AUG', 'Total', 'DEC', 'FEB', 'JAN', 'JUL', 'JUN', 'MAR', 'MAY', 'NOV', 'OCT','SEP', 'Cattle Lost (in Nos.)', 'Cropped areas affected (in lakh ha)', 'House damaged (in Nos.)', 'Lives Lost (in Nos.)']]
plt.bar(MasterData['Year'],Y, align='center', alpha=0.5)
plt.show()
