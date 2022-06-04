import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix

df = pd.read_csv("titanic.csv")
df_age = df[['Age', 'Survived']]
df_age.dropna(inplace=True)
df_age.reset_index(drop=True, inplace=True)

learning_rate=.1
gradient_descent=True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



X = np.array(df_age['Age'])


y = df_age['Survived']
y = np.array(y.tolist())


# INICIALIZANDO PARÂMETROS
def inicia_parametros(n_features):
    n_features = 1
    # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
    limit = 1 / math.sqrt(n_features)
    initialize_parameters = np.random.uniform(-limit, limit, (n_features,))
    return initialize_parameters

parametros_iniciados = [inicia_parametros(1) for i in range(len(X))]
parametros = [parametros_iniciados[i][0] for i in range(len(parametros_iniciados))]


iteracoes = 200

for i in range(iteracoes):
    # FOR FAZENDO UMA NOVA PREDIÇÃO
    y_pred = [sigmoid(parametros[i]) for i in range(len(parametros))]
    y_pred = [y_pred[i] for i in range(len(y_pred))]
    y_pred = np.array(y_pred)

    parametros += learning_rate * (y - y_pred) * X


y_pred = y_pred > 0.5


confusion_matrix(y, y_pred)

