import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


# membaca dataset dan mengubahnya menjadi dataframe
data = pd.read_csv('./data/Salary_Data.csv')
 
# memisahkan atribut dan label
X = data['YearsExperience']
y = data['Salary']
 
# mengubah bentuk atribut
X = np.array(X)
X = X.reshape(-1,1)


 
# membangun model dengan parameter C, gamma, dan kernel
model = SVR()
parameters = {
    'kernel': ['rbf'],
    'C':     [1000, 10000, 100000],
    'gamma': [0.5, 0.05,0.005]
}
grid_search = GridSearchCV(model, parameters)
 
# melatih model dengan fungsi fit
grid_search.fit(X,y)

# menampilkan parameter terbaik dari objek grid_search
print(grid_search.best_params_)

# membuat model SVM baru dengan parameter terbaik hasil grid search
model_baru  = SVR(C=100000, gamma=0.005, kernel='rbf')
model_baru.fit(X,y)


plt.scatter(X, y)
plt.plot(X, model_baru.predict(X))

plt.show()