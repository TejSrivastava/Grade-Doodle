import pandas as pd
import numpy as np

import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:\\Users\\Tejas\\student\\student-mat.csv", sep=";")

print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

predict = "G3"

X = np.array(data.drop([predict], axis=1))
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

print("Coefficient: \n" ,linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions=linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
    








 