import numpy as num
import matplotlib.pyplot as mpl
import pandas as pd

data = pd.read_csv('data.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=num.nan, strategy='mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)

