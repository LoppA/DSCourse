import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from numpy import mean, std

def map_gender(data):
    ret = data.copy()
    m = {'male' : 0, 'female' : 1}
    ret['Sex'] = ret['Sex'].str.lower().map(m)
    return ret

def classifica(data):
    data = map_gender(data)
    target = 1
    columns = [2, 4, 5, 6, 7, 9]
    reg_clas(data, columns, target, regression=False)

def reg_clas(data, columns, target, regression=False):
    """
    Executa classificação ou regressão do conjunto de dados passado
    ---------------------------------------------------------------
    data:       DataFrame. Conjunto de dados
    columns:    Lista de inteiros. Índice das colunas utilizadas no treinamento e teste
    target:     Inteiro. Índice da coluna alvo
    regression: Boleano. True para realizar uma regressão e False para classificação
    ---------------------------------------------------------------
    Realiza a classificação/regressão em 4 modelos (SVM com kernel linear, 
    SVM com kernel polinomial de grau 3, Árvore de decisão, e knn com k=5)
    Plot o gráfico de desempenho para cada classificador/regressor.
    Retorna um dicionário com os classificadores/regressores treinados e as medidas de desempenho
    """
    
    # inicializa os modelos com os parâmetros solicitados
    if regression:
        svm_l = SVR(C=10*len(data), kernel='linear', cache_size=500, max_iter=1e6)
        svm_n = SVR(C=10*len(data), kernel='poly', degree=3, gamma=1, coef0=1, cache_size=500, max_iter=1e6)
        dt = DecisionTreeRegressor(criterion='mse', splitter='best', min_samples_split=int(len(data)*0.05))
        knn = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='kd_tree')
    else:
        svm_l = SVC(C=10*len(data), kernel='linear', cache_size=500, max_iter=1e6)
        svm_n = SVC(C=10*len(data), kernel='poly', degree=3, gamma=1, coef0=1, cache_size=500, max_iter=1e6)
        dt = DecisionTreeClassifier(criterion='gini', splitter='best', min_samples_split=int(len(data)*0.1))
        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='kd_tree')
    
    clfs = [svm_l, svm_n, dt, knn]
    clfs_names = ['svm_linear', 'svm_poly', 'dt', 'knn']
    
    # prepara validação cruzada
    # faz divisão do dataset em 5 partes
    cv = KFold(n_splits=5, shuffle=True)
    
    # itera para cada classificador fazendo treino e teste
    results = {'svm_linear':[], 'svm_poly':[], 'dt':[], 'knn':[]}
    for c, c_name in zip(clfs, clfs_names):
        for train_index, test_index in cv.split(data):
            
            # separa conjunto de treino e de teste
            x_train, y_train = data.iloc[train_index, columns], data.iloc[train_index, target]
            x_test, y_test = data.iloc[test_index, columns], data.iloc[test_index, target]
            
            # faz o treino do modelo
            clf = c.fit(X=x_train, y=y_train)
            
            # realiza predição no conjunto de teste e salva o resultado
            if regression:
                results[c_name].append( mean_squared_error(clf.predict(x_test), y_test) )
            else:
                results[c_name].append( clf.score(x_test, y_test) )
    
    # faz o plot de desempenho dos classificadores/regressores
    plt.figure(figsize=(8,8))
    plt.bar(range(1, len(clfs)+1), [mean(results[name]) for name in clfs_names], 
                                yerr=[std(results[name]) for name in clfs_names])
    plt.xticks(range(1, len(clfs)+1), clfs_names, rotation=45)
    title = 'Desempenho dos regressores - erro quadrático médio' if regression else \
            'Desempenho dos classificadores - acurácia'
    plt.title(title)
    plt.show()
    
    return {'results': results, 'clfs':clfs}

'''
titanic = pd.read_csv("./datasets/titanic/train.csv")

print(titanic.isna().sum() / len(titanic))

titanic.drop(['Cabin'], inplace=True, axis=1)

titanic_dropna = titanic.copy()
titanic_dropna.dropna(axis=0, how='any', inplace=True)
print(titanic_dropna.isna().sum())

titanic_fill_zero = titanic.copy()
titanic_fill_zero.fillna(value=0, axis=0, inplace=True)
print(titanic_fill_zero.isna().sum())

print(titanic.mode(axis=0).iloc[0, :])

titanic_fill_mean = titanic.copy()
titanic_fill_mean.fillna(value=titanic.mean(), axis=0, inplace=True)
titanic_fill_mean.fillna(value=titanic.mode(axis=0).iloc[0, :], axis=0, inplace=True)
print(titanic_fill_mean.isna().sum())

titanic_fill_median = titanic.copy()
titanic_fill_median.fillna(value=titanic.median(), axis=0, inplace=True)
titanic_fill_median.fillna(value=titanic.mode(axis=0).iloc[0, :], axis=0, inplace=True)
print(titanic_fill_median.isna().sum())

classifica(titanic_dropna)

classifica(titanic_fill_zero)

classifica(titanic_fill_mean)

classifica(titanic_fill_median)
'''

inst_ens = pd.read_csv("./datasets/instituicoes_ensino_basico/CADASTRO_MATRICULAS_REGIAO_SUDESTE_SP_2012.csv", encoding="ISO-8859-1", sep=";", engine="python", header=11, skipfooter=2)

columns = ['REDE', 'Localizacao', 'NUM_SALAS_EXISTENTES', 'NUM_FUNCIONARIOS', 'NUM_COMPUTADORES']
data = inst_ens[columns]
print(data.isna().sum() / len(inst_ens))
