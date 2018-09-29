import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt

inst_ens = pd.read_csv("./datasets/instituicoes_ensino_basico/CADASTRO_MATRICULAS_REGIAO_SUDESTE_SP_2012.csv", 
                               encoding="ISO-8859-1", sep=";", engine="python", header=11, skipfooter=2)

from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from numpy import mean, std


def classificacao(data, columns, target, weights):
    """
    Executa classificação do conjunto de dados passado
    ---------------------------------------------------------------
    data:       DataFrame. Conjunto de dados
    columns:    Lista de inteiros. Índice das colunas utilizadas no treinamento e teste
    target:     Inteiro. Índice da coluna alvo
    weights:    Dicionário. Pesos das classes, cada chave do dicionário é a classe e seu valor a probabilidade
    ---------------------------------------------------------------
    Realiza a classificação em 3 modelos (SVM com kernel linear, 
    SVM com kernel polinomial de grau 3, Árvore de decisão)
    Plot o gráfico de desempenho para cada classificador.
    Retorna um dicionário com os classificadores treinados e as medidas de desempenho
    """
    
    # inicializa os modelos com os parâmetros solicitados
    svm_l = SVC(C=10*len(data), kernel='linear', cache_size=500, max_iter=1e6, class_weight=weights)
    svm_n = SVC(C=10*len(data), kernel='poly', degree=3, gamma=1, coef0=1, cache_size=500, max_iter=1e6, class_weight=weights)
    dt = DecisionTreeClassifier(criterion='gini', splitter='best', min_samples_split=int(len(data)*0.1), class_weight=weights)
    
    clfs = [svm_l, svm_n, dt]
    clfs_names = ['svm_linear', 'svm_poly', 'dt']
    
    # prepara validação cruzada
    # faz divisão do dataset em 5 partes
    cv = KFold(n_splits=5, shuffle=True)
    
    # itera para cada classificador fazendo treino e teste
    results = {'svm_linear':[], 'svm_poly':[], 'dt':[]}
    for c, c_name in zip(clfs, clfs_names):
        for train_index, test_index in cv.split(data):
            
            # separa conjunto de treino e de teste
            x_train, y_train = data.iloc[train_index, columns], data.iloc[train_index, target]
            x_test, y_test = data.iloc[test_index, columns], data.iloc[test_index, target]
            
            # faz o treino do modelo
            clf = c.fit(X=x_train, y=y_train)
            
            # realiza predição no conjunto de teste e salva o resultado
            results[c_name].append( clf.score(x_test, y_test) )
    
    # faz o plot de desempenho dos classificadores
    plt.figure(figsize=(8,8))
    plt.bar(range(1, len(clfs)+1), [mean(results[name]) for name in clfs_names], 
                                yerr=[std(results[name]) for name in clfs_names])
    plt.xticks(range(1, len(clfs)+1), clfs_names, rotation=45)
    title = 'Desempenho dos classificadores - acurácia'
    plt.title(title)
    plt.show()
    
    return {'results': results, 'clfs':clfs}

def classifica(data, target, colums, weights):
    print("Target: ", data.columns[target], "\nColumns: ", data.columns[columns], "\n Weights: ", weights)
    data_dropna = data.copy()
    data_dropna.dropna(axis=0, how='any', inplace=True)
    data_dropna = data_dropna.sample(1000)
    classificacao(data_dropna, columns, target, weights)

columns = ['REDE', 'Localizacao']

sub_inst_ens = [inst_ens.loc[inst_ens.Localizacao == c, :].sample(n=1908, replace=True) for c in inst_ens.Localizacao.unique()]
print('\n\nFixando o número de exemplos por classe em 1908')
sub_inst_ens = pd.concat(sub_inst_ens)

sup_inst_ens = [inst_ens.loc[inst_ens.Localizacao == c, :].sample(n=29151, replace=True) for c in inst_ens.Localizacao.unique()]
print('\n\nFixando o número de exemplos por classe em 29151')
sup_inst_ens = pd.concat(sup_inst_ens)

columns = ['Localizacao', 'NUM_SALAS_EXISTENTES', 'NUM_FUNCIONARIOS', 'NUM_COMPUTADORES']
data = sub_inst_ens[columns]

'''
target = 0
columns = [1, 2, 3]
weights={'Rural':0.5, 'Urbana':0.5}
classifica(data, target, columns, weights)
'''

'''
target = 0
columns = [1, 2, 3]
weights={'Rural':29151/(1908+29151), 'Urbana':1908/(1908+29151)}
classifica(data, target, columns, weights)
'''

inst_ens_2 = inst_ens.copy()
inst_ens_2["Rede_Local"] = inst_ens_2["REDE"].map(str) + "_" + inst_ens_2["Localizacao"]

m_map = {'privada_rural' : 1, 'privada_urbana': 0, 'pública_rural' : 0, 'pública_urbana': 0}
inst_ens_2["Privada_Rural"] = inst_ens_2["Rede_Local"].str.lower().map(m_map)

m_map = {'privada_rural' : 0, 'privada_urbana': 1, 'pública_rural' : 0, 'pública_urbana': 0}
inst_ens_2["Privada_Urbana"] = inst_ens_2["Rede_Local"].str.lower().map(m_map)

m_map = {'privada_rural' : 0, 'privada_urbana': 0, 'pública_rural' : 1, 'pública_urbana': 0}
inst_ens_2["Pública_Rural"] = inst_ens_2["Rede_Local"].str.lower().map(m_map)

m_map = {'privada_rural' : 0, 'privada_urbana': 0, 'pública_rural' : 0, 'pública_urbana': 1}
inst_ens_2["Pública_Urbana"] = inst_ens_2["Rede_Local"].str.lower().map(m_map)

inst_ens_2.drop(['Rede_Local'], inplace=True, axis=1)

inst_ens_2["NUM_SALAS_EXISTENTES"] = pd.cut(inst_ens_2["NUM_SALAS_EXISTENTES"], bins=[0, 10, 20, 200], include_lowest=True, labels=['baixo', 'medio', 'alto'])

columns = ['REDE', 'Localizacao', 'Privada_Rural', 'Privada_Urbana', 'Pública_Rural', 'Pública_Urbana', 'NUM_SALAS_EXISTENTES', 'NUM_FUNCIONARIOS', 'NUM_COMPUTADORES']
print(inst_ens_2[columns])

inst_ens_2 = inst_ens_2[columns]
inst_ens_2.dropna(axis=0, how='any', inplace=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
inst_ens_2[['NUM_FUNCIONARIOS']] = scaler.fit_transform(inst_ens_2[['NUM_FUNCIONARIOS']])

col = inst_ens_2['NUM_COMPUTADORES']
inst_ens_2['NUM_COMPUTADORES'] = (col - col.mean()) / col.std(ddof=0)

print(inst_ens_2)
