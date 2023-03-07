import numpy as num
import matplotlib.pyplot as mpl
import pandas as pd

# odczytanie tabeli z pliku
data = pd.read_csv('data.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# podmiana pustych wartości na średnią arytmetyczna
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=num.nan, strategy='mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#print(x)
# zmienianie danych na liczby encodowane
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = num.array(ct.fit_transform(x))
#print(x)

#zmiana wartości Yes i No na wartości liczbowe
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#print(y)

#train i test code
from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# print(x_train)
# print(x_test)
# print(y_test)
# print(y_train)

#standaryzacja
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
# print(x_train)
# print(x_test)