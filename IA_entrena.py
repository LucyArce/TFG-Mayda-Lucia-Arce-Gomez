import os
import csv
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve


from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor


def EntrenaIA(sal):
    file_path = r'C:\Users\mlarc\Desktop\RoboDK\Samples\Base de datos\Base de datos-csv'
    file_nametra = 'DataBaseTrayectoria.csv'
    csv_file_path_tra = os.path.join(file_path, file_nametra)

    rdata = pd.read_csv(csv_file_path_tra)
    X = rdata[['x', 'y']]
    if sal == 1:
        y_v1 = rdata['v1']
        y_v2 = rdata['v2']
        return X, y_v1, y_v2

    if sal == 2:
        y = rdata[['v1', 'v2']]
        return X, y


###Cuarta Lección (o Capítulo): Modelado del Comportamiento del Data Set
#8.2.1 Regresión polinómica
def Poli(Xin, enTR):
    if enTR == 0:
        model_v1 = joblib.load('model_v1.pkl')
        model_v2 = joblib.load('model_v2.pkl')
        poly = joblib.load('poly_transformer.pkl')
        # Realizar predicciones
        pred_v1 = model_v1.predict(Xin)
        pred_v2 = model_v2.predict(Xin)
    
        return pred_v1, pred_v2

    else:
        X, y_v1, y_v2 = EntrenaIA(1)
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train_v1, y_test_v1 = train_test_split(X, y_v1, test_size=0.2, random_state=25)
        _, _, y_train_v2, y_test_v2 = train_test_split(X, y_v2, test_size=0.2, random_state=25)

        # Crear transformador polinómico de grado 2
        poly = PolynomialFeatures(degree=2)

        # Transformar las características de entrenamiento y prueba
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Crear y entrenar el modelo para v1
        model_v1 = LinearRegression()
        model_v1.fit(X_train_poly, y_train_v1)

        # Crear y entrenar el modelo para v2
        model_v2 = LinearRegression()
        model_v2.fit(X_train_poly, y_train_v2)

        # Guardar los modelos y el transformador polinómico
        joblib.dump(model_v1, 'POL_model_v1.pkl')
        joblib.dump(model_v2, 'POL_model_v2.pkl')
        joblib.dump(poly, 'poly_transformer.pkl')

        return model_v1, model_v2

    


#8.2.2 Árboles de decisión: Enrenamiento de dos árboles
def Arb2(Xin, enTR):
    if enTR == 0:
        model_v1 = joblib.load('ARB2_model_v1.pkl')
        model_v2 = joblib.load('ARB2_model_v2.pkl')
        # Realizar predicciones
        pred_v1 = model_v1.predict(Xin)
        pred_v2 = model_v2.predict(Xin)
    
        return pred_v1, pred_v2
        # Predicciones
        y_test_pred_v1 = model_v1.predict(X_test)
        y_test_pred_v2 = model_v2.predict(X_test)
        v1 = y_test_pred_v1
        v2 = y_test_pred_v2

    else:
        X, y_v1, y_v2 = EntrenaIA(1)
        # Dividir los datos en conjunto de entrenamiento y conjunto de prueba para cada salida
        X_train, X_test, y_train_v1, y_test_v1 = train_test_split(X, y_v1, test_size=0.2, random_state=25)
        _, _, y_train_v2, y_test_v2 = train_test_split(X, y_v2, test_size=0.2, random_state=25)

        # Crear el modelo de árbol de decisión para cada salida
        model_v1 = DecisionTreeRegressor(random_state=50)
        model_v2 = DecisionTreeRegressor(random_state=50)

        # Entrenar los modelos
        model_v1.fit(X_train, y_train_v1)
        model_v2.fit(X_train, y_train_v2)

        # Guardar los modelos
        joblib.dump(model_v1, 'ARB2_model_v1.pkl')
        joblib.dump(model_v2, 'ARB2_model_v2.pkl')
        return model_v1, model_v2



#8.2.2 Árboles de decisión: Multisalida
def Multi(Xin, enTR):
    if enTR == 0:
        model = joblib.load('MultiARB_model.pkl')
        # Predicciones en el conjunto de prueba
        y_test_pred = model.predict(Xin)
        v1 = y_test_pred[:, 0]
        v2 = y_test_pred[:, 1]
        return v1, v2
    else:
        X, y = EntrenaIA(2)
        # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

        # Crear el modelo de árbol de decisión para múltiples salidas
        model = DecisionTreeRegressor(random_state=50)

        # Entrenar el modelo
        model.fit(X_train, y_train)

        # Guardar modelo
        joblib.dump(model, 'MultiARB_model.pkl')
        return model


#8.2.3 K-NNs
def kNN(Xin, enTR):
    if enTR == 0:
        model_v1 = joblib.load('KNN_model_v1.pkl')
        model_v2 = joblib.load('KNN_model_v2.pkl')
        y_pred_v1 = model_v1.predict(Xin)
        y_pred_v2 = model_v2.predict(Xin)
        v1 = y_pred_v1
        v2 = y_pred_v2
        return v1, v2

    else:
        X, y_v1, y_v2 = EntrenaIA(1)
        X_train, X_test, y_train_v1, y_test_v1 = train_test_split(X, y_v1, test_size=0.2, random_state=42)
        _, _, y_train_v2, y_test_v2 = train_test_split(X, y_v2, test_size=0.2, random_state=42)

        model_v1 = KNeighborsRegressor(n_neighbors=10)
        model_v1.fit(X_train, y_train_v1)
        

        model_v2 = KNeighborsRegressor(n_neighbors=10)
        model_v2.fit(X_train, y_train_v2)

        # Guardar los modelos
        joblib.dump(model_v1, 'KNN_model_v1.pkl')
        joblib.dump(model_v2, 'KNN_model_v2.pkl')
        return model_v1, model_v2
    

    



#8.2.4 Redes neuronales
def RedNeu(Xin, enTR):
    if enTR == 0:
        model = joblib.load('RED_model.pkl')
        # Hacer predicciones
        predictions = model.predict(X_test)
        v1, v2 = predictions
        return v1, v2

    else:
        X, y = EntrenaIA(2)
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

        # Crear y entrenar el modelo MLP
        model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000)
        model.fit(X_train, y_train)

        # Guardar modelo
        joblib.dump(model, 'RED_model.pkl')
        return model




def modelo_usar(x, y, Mod):
    xtest = np.array([[x, y]])
    if Mod == 1:
        poly = PolynomialFeatures(degree=2)
        X_test_poly = poly.fit_transform(xtest)
        v1, v2 = Poli(X_test_poly, 0)
    elif Mod == 2:
        v1, v2 = Arb2(xtest, 0)
    elif Mod == 3:
        v1, v2 = Multi(xtest, 0)
    elif Mod == 4:
        v1, v2 = kNN(xtest, 0)
    elif Mod == 5:
        v1, v2 = Multi(xtest, 0)
    else:
        v1 = 10000
        v2 = 10000
    #print("Modelo", Mod)
    #print("Predicción v1", v1)
    #print("Predicción v2", v2)
    return v1, v2

#RedNeu([100, 230], 1)
#modelo_usar(100, 230, 5)