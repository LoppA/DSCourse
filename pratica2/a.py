import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt

def calc(values):
    med = 0.0
    quant = 0.0
    for i in range(values.shape[0]):
        if(np.isnan(values[i]) == False):
            quant += 1.0
            med += values[i]
    med /= quant

    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    for i in range(values.shape[0]):
        if(np.isnan(values[i]) == False):
            m2 += (values[i] - med)**2
            m3 += (values[i] - med)**3
            m4 += (values[i] - med)**4
    m2 /= quant
    m3 /= quant
    m4 /= quant

    print('Momento 1: ', med, '| Momento 2:', m2, '| Momento 3:', m3, '| Momento 4:', m4)

def momento(nome, name1, name2, class1, class2):
    print(nome, ' ', name1, '-> ', sep='', end='')
    calc(class1)
    print(nome, ' ', name2, '-> ', sep='', end='')
    calc(class2)

'''
titanic = pd.read_csv("./datasets/titanic/train.csv")

print(titanic.columns[[1,5,6,9]])

titanic.boxplot(column=titanic.columns[[5,6,9]].tolist(), by='Survived')
plt.show()

v_sobrevivente = np.ndarray.astype((titanic["Survived"] == 1).values, dtype=np.bool)
sobrevivente1 = titanic[v_sobrevivente]
sobrevivente0 = titanic[np.logical_not(v_sobrevivente)]

print()
momento('Age', 'Survived=0', 'Survived=1', sobrevivente0.Age.values, sobrevivente1.Age.values)
print()
momento('SibSp', 'Survived=0', 'Survived=1', sobrevivente0.SibSp.values, sobrevivente1.SibSp.values)
print()
momento('Fare', 'Survived=0', 'Survived=1', sobrevivente0.Fare.values, sobrevivente1.Fare.values)
print()

print(titanic[titanic.Age > 60])

print(titanic[titanic.Fare > 50.0])

print(titanic[titanic.SibSp > 2])

print("Age: ", titanic.Age.max() - titanic.Age.min())
print("Fare: ", titanic.Fare.max() - titanic.Fare.min())
print("SibSp: ", titanic.SibSp.max() - titanic.SibSp.min())
'''


inst_ens = pd.read_csv("./datasets/instituicoes_ensino_basico/CADASTRO_MATRICULAS_REGIAO_SUDESTE_SP_2012.csv", 
                               encoding="ISO-8859-1", sep=";", engine="python", header=11, skipfooter=2)

for i in range(inst_ens.columns.shape[0]):
    print(inst_ens.columns[i])

inst_ens.boxplot(column=['NUM_SALAS_EXISTENTES', 'NUM_FUNCIONARIOS'], by='REDE')
plt.show()

inst_ens.boxplot(column=['NUM_COMPUTADORES'], by='Localizacao')
plt.show()

print("NUM_SALAS_EXISTENTES: ", inst_ens.NUM_SALAS_EXISTENTES.max()- inst_ens.NUM_SALAS_EXISTENTES.min())
print("NUM_FUNCIONARIOS: ", inst_ens.NUM_FUNCIONARIOS.max()- inst_ens.NUM_FUNCIONARIOS.min())
print("NUM_COMPUTADORES: ", inst_ens.NUM_COMPUTADORES.max()- inst_ens.NUM_COMPUTADORES.min())

v_privada = np.ndarray.astype((inst_ens["REDE"] == 'Privada').values, dtype=np.bool)
privada = inst_ens[v_privada]
publica = inst_ens[np.logical_not(v_privada)]

v_rural = np.ndarray.astype((inst_ens["Localizacao"] == 'Rural').values, dtype=np.bool)
rural = inst_ens[v_rural]
urbana = inst_ens[np.logical_not(v_rural)]

print()
momento('NUM_SALAS_EXISTENTES', 'Rede privada', 'Rede publica', privada.NUM_SALAS_EXISTENTES.values, publica.NUM_SALAS_EXISTENTES.values)
print()
momento('NUM_FUNCIONARIOS', 'Rede privada', 'Rede publica', privada.NUM_FUNCIONARIOS.values, publica.NUM_FUNCIONARIOS.values)
print()
momento('NUM_COMPUTADORES', 'Rede rural', 'Rede urbana', rural.NUM_COMPUTADORES.values, urbana.NUM_COMPUTADORES.values)
print()
