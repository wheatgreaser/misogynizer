import pandas as pd
from sklearn import linear_model
import numpy

data = [["Man","Cool"], ["Female", "Bad"],["Man", "Cool"], ["Man","Cool"], ["Female", "Bad"],  ["Man","Bad"], ["Female", "Cool"],["Man","Cool"], ["Female", "Cool"]]

df = pd.DataFrame(data, columns=["Gender", "Attribute"])


oneHotEncodedData = pd.get_dummies(df, columns = ["Gender", "Attribute"])
print(oneHotEncodedData)

linreg = linear_model.LinearRegression()

X = numpy.array(oneHotEncodedData["Gender_Female"]).reshape(-1, 1)
Y = numpy.array(oneHotEncodedData["Attribute_Cool"])

linreg.fit(X, Y)
print(linreg.predict(numpy.array([False]).reshape(-1, 1)))