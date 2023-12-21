import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 

 
# membaca dataset dan mengubahnya menjadi dataframe
df = pd.read_csv('./data/Social_Network_Ads.csv')

# df.info()
# print(df.head())


# drop kolom yang tidak diperlukan
data = df.drop(columns=['User ID'])
 
# jalankan proses one-hot encoding dengan pd.get_dummies()
data = pd.get_dummies(data)
# print(data)

# pisahkan atribut dan label
predictions = ['Age' , 'EstimatedSalary' , 'Gender_Female' , 'Gender_Male']
X = data[predictions]
y = data['Purchased'] 

# lakukan normalisasi terhadap data yang kita miliki
scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)
scaled_data = pd.DataFrame(scaled_data, columns= X.columns)
# print(scaled_data.head())

# bagi data menjadi train dan test untuk setiap atribut dan label
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2, random_state=1)

linier_regresion = LogisticRegression()
linier_regresion.fit(X_train, y_train)

# uji akurasi model
accuration = linier_regresion.score(X_test, y_test)
print(accuration)