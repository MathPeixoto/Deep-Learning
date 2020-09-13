import pandas as pd
from tensorflow.keras.layers import Dense, Input, Dropout # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Model # atualizado: tensorflow==2.0.0-beta1
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import backend as k
from sklearn.model_selection import cross_val_score

base = pd.read_csv('games.csv')
base = base.drop('Other_Sales', axis = 1)
base = base.drop('Developer', axis = 1)

base = base.dropna(axis = 0)
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]
base = base.loc[base['Global_Sales'] > 1]

base = base.drop('Name', axis = 1)

previsores = base.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11]].values
vendas_gl = base.iloc[:, 7].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

def criar_rede():
    k.clear_session()
    camada_entrada = Input(shape=(101,))
    camada_oculta1 = Dense(units = 51, activation = 'relu')(camada_entrada)
    Dropout(0.1)(camada_oculta1)
    camada_oculta2 = Dense(units = 51, activation = 'relu')(camada_oculta1)
    camada_saida1 = Dense(units = 1, activation = 'linear')(camada_oculta2)
    regressor = Model(inputs = camada_entrada, outputs = [camada_saida1])
    regressor.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
    return regressor

regressor = KerasRegressor(build_fn = criar_rede, epochs = 10000, batch_size = 200)

resultados = cross_val_score(estimator = regressor,
                             X = previsores, y = vendas_gl,
                             cv = 8, scoring = 'neg_mean_squared_error')

media = resultados.mean()
desvio = resultados.std()

regressor.fit(previsores, vendas_gl)
previsao_gl = regressor.predict(previsores)
