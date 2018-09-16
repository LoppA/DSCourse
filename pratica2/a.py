import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt

titanic = pd.read_csv("./datasets/titanic/train.csv")

print(titanic.columns[[1,5,6,9]])

#titanic.boxplot(column=titanic.columns[[5,6,9]].tolist(), by='Survived')
#plt.show()

# print(titanic[titanic.Age > 60])

#print(titanic[titanic.Fare > 50.0])

#print(titanic[titanic.SibSp > 2])

print("Age: ", titanic.Age.max() - titanic.Age.min())
print("Fare: ", titanic.Fare.max() - titanic.Fare.min())
print("SibSp: ", titanic.SibSp.max() - titanic.SibSp.min())
