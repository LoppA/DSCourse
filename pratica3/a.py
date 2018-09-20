import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
import seaborn as sns

def tira_nan(x):
    return x[~np.isnan(x)]

def clear(x, col):
    return x[~np.isnan(x[col])]

def hist(x, y, labelx, bins):
    plt.xlabel(labelx, fontsize=14)  
    plt.ylabel("Occurrences", fontsize=14)
    plt.hist(x, bins=bins, fc=(0, 0, 1, 0.5))
    plt.hist(y, bins=bins, fc=(0, 1, 0, 0.5))
    plt.tight_layout()
    labels = ["Didn't survive", "Survived"]
    plt.legend(labels)
    plt.show()

'''
titanic = pd.read_csv("./datasets/titanic/train.csv")

v_sobrevivente = np.ndarray.astype((titanic["Survived"] == 1).values, dtype=np.bool)
sobrevivente1 = titanic[v_sobrevivente]
sobrevivente0 = titanic[np.logical_not(v_sobrevivente)]
'''

#hist(tira_nan(sobrevivente0.Age.values), tira_nan(sobrevivente1.Age.values), "Age", 20)

#hist(tira_nan(sobrevivente0.SibSp.values), tira_nan(sobrevivente1.SibSp.values), "Relatives aboard", 8)

#hist(tira_nan(sobrevivente0.Fare.values), tira_nan(sobrevivente1.Fare.values), "Fare", 20)

'''
titanic0 = titanic[v_sobrevivente]
titanic1 = titanic[~v_sobrevivente]

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title("All")
cov = titanic.iloc[:, [5,6,9]].cov()
sns.heatmap(cov)

plt.subplot(1, 3, 2)
plt.title("Didn't survive")
cov = titanic0.iloc[:, [5,6,9]].cov()
sns.heatmap(cov)

plt.subplot(1, 3, 3)
plt.title("Survived")
cov = titanic1.iloc[:, [5,6,9]].cov()
sns.heatmap(cov)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title("All")
cor = titanic.iloc[:, [5,6,9]].corr()
sns.heatmap(cor)

plt.subplot(1, 3, 2)
plt.title("Didn't survive")
cor = titanic0.iloc[:, [5,6,9]].corr()
sns.heatmap(cor)

plt.subplot(1, 3, 3)
plt.title("Survived")
cor = titanic1.iloc[:, [5,6,9]].corr()
sns.heatmap(cor)

plt.tight_layout()
plt.show()
'''

'''
columns = ["Age", "SibSp", "Fare"]
titanic = clear(titanic, "Age")
titanic = clear(titanic, "SibSp")
titanic = clear(titanic, "Fare")
sns.pairplot(data=titanic, vars=columns, hue='Survived')
plt.show()
'''

inst_ens = pd.read_csv("./datasets/instituicoes_ensino_basico/CADASTRO_MATRICULAS_REGIAO_SUDESTE_SP_2012.csv", encoding="ISO-8859-1", sep=";", engine="python", header=11, skipfooter=2)

v_privada = np.ndarray.astype((inst_ens["REDE"] == 'Privada').values, dtype=np.bool)
privada = inst_ens[v_privada]
publica = inst_ens[np.logical_not(v_privada)]

v_rural = np.ndarray.astype((inst_ens["Localizacao"] == 'Rural').values, dtype=np.bool)
rural = inst_ens[v_rural]
urbana = inst_ens[np.logical_not(v_rural)]
'''
for i in range(inst_ens.columns.shape[0]):
    print(i, inst_ens.columns[i])
'''
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Todas")
cov = inst_ens.iloc[:, [80, 86, 110]].cov()
sns.heatmap(cov)

plt.subplot(1, 3, 2)
plt.title("Privada")
cov = privada.iloc[:, [80, 86, 110]].cov()
sns.heatmap(cov)

plt.subplot(1, 3, 3)
plt.title("Publica")
cov = publica.iloc[:, [80, 86, 110]].cov()
sns.heatmap(cov)

plt.tight_layout()
plt.show()
