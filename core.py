#
import warnings

# Graficos
import matplotlib.pyplot as plt

# Operacoes matematicas e em matriz
import numpy as np
import pandas as pd
from math import sqrt
# Algoritmos
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
# Metricas
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# Suppress warnings
warnings.filterwarnings("ignore")


#
dataframe = pd.read_csv("newData.csv", header=0)
dataframe = dataframe.dropna()
dataframe = dataframe.drop_duplicates()
dataframe = dataframe.drop(dataframe.columns[0], axis=1)


print(dataframe.columns)

	
y_dataframe = dataframe[['Price']]
X_dataframe = dataframe.loc[:, dataframe.columns != 'Price']

algs = [DecisionTreeRegressor(random_state=0, criterion='mse'),
	KNeighborsRegressor(),
        MLPRegressor()]

media_rmses = []
media_maes = []

#Ajustes MLP
MLPRegressor(hidden_layer_sizes=(3), 
                  activation='tanh', solver='lbfgs')


for n in range(1):
    # , random_state=i+1
    train, test = train_test_split(dataframe, test_size=0.20, random_state=n+1)

    X_train = train.loc[:, dataframe.columns != 'Price']
    y_train = train['Price']
    X_test = test.loc[:, dataframe.columns != 'Price']
    y_test = test['Price']

    for model in algs:
        alg = model
        print("Treinando com {}".format(model))

        mdl = alg.fit(X_train, y_train)
        mdl_predict = mdl.predict(X_test)

        rmse = sqrt(mean_squared_error(y_test, mdl_predict))
        mae = mean_absolute_error(y_test, mdl_predict)

        print("RMSE (Raiz do Erro Quadratico Medio): ", rmse)
        print("MAE (Erro Absoluto Medio): ", mae)
        media_rmses.append(rmse)
        media_maes.append(mae)


for count in range(0, 3):
    name = ''
    vec_rmse = []
    vec_mae = []

    if count == 0:
        name = 'DecisionTree'
    elif count == 1:
        name = 'KNN'
    else:
        name = 'MLP'

    for i in range(count, len(media_rmses), 3):
        vec_rmse.append(media_rmses[i])
        vec_mae.append(media_maes[i])

    print("Media do RMSE do " + name + ": " + str(np.mean(vec_rmse)))
    print("Media do MAE do " + name + ": " + str(np.mean(vec_mae)))



