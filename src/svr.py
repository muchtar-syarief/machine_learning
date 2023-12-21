import numpy as np
import pandas as pd
 
import matplotlib.pyplot as plt
from sklearn.svm import SVR


# membaca dataset dan mengubahnya menjadi dataframe
data = pd.read_csv('./data/Salary_Data.csv')

data.info()

data.head()

 
# memisahkan atribut dan label
X = data['YearsExperience']
y = data['Salary']
 
# mengubah bentuk atribut
X = np.array(X)
X = X[:,np.newaxis]

 
# membangun model dengan parameter C, gamma, dan kernel
model  = SVR(C=1000, gamma=0.05, kernel='rbf')
 
# melatih model dengan fungsi fit
model.fit(X,y)

 
# memvisualisasikan model
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.show()