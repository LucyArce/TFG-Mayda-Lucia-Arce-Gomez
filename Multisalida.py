import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import os
import csv
import pandas as pd

file_path = r'E:\RoboDK\Samples\Base de datos\DataSet1'
file_namemin = 'DataBaseMinimos.csv'
csv_file_path_min = os.path.join(file_path, file_namemin)

rdata = pd.read_csv(csv_file_path_min)

# Separar características (X) y variables objetivo (y)
xx = rdata[['x', 'y']]
yy = rdata[['v1', 'v2']]  # Notar que y ahora es un DataFrame con dos columnas

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2, random_state=25)

# Crear el modelo de árbol de decisión para múltiples salidas
tree_reg = DecisionTreeRegressor(random_state=50)

# Entrenar el modelo
tree_reg.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_test_pred = tree_reg.predict(X_test)

# Calcular el error cuadrático medio para cada salida
mse_test_v1 = mean_squared_error(y_test['v1'], y_test_pred[:, 0])
mse_test_v2 = mean_squared_error(y_test['v2'], y_test_pred[:, 1])
print("Error cuadrático medio en conjunto de prueba para v1:", mse_test_v1)
print("Error cuadrático medio en conjunto de prueba para v2:", mse_test_v2)

from sklearn.metrics import r2_score

r2v1 = r2_score(y_test['v1'], y_test_pred[:, 0])
print("Coeficiente de Determinación (R²):", r2v1)
r2v2 = r2_score(y_test['v2'], y_test_pred[:, 1])
print("Coeficiente de Determinación (R²):", r2v2)
'''
plt.figure(figsize=(10, 6))
plt.scatter(xx['x'], xx['y'], color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Diagrama de dispersión de las entradas')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(range(len(yy['v1'])), yy['v1'], color='green', label='v1')
plt.scatter(range(len(yy['v2'])), yy['v2'], color='orange', label='v2')
plt.xlabel('Índice de la muestra')
plt.ylabel('Valores de salida')
plt.title('Diagrama de dispersión de las salidas')
plt.legend()
plt.show()
'''
# Graficar predicciones para la salida v1
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_test['x'], y_test['v1'], color='blue', label='Datos reales')
plt.scatter(X_test['x'], y_test_pred[:, 0], color='red', label='Predicciones')
plt.xlabel('x')
plt.ylabel('v1')
plt.title('Predicciones vs Datos reales - Salida v1')
plt.legend()


# Graficar predicciones para la salida v2
plt.subplot(1, 2, 2)
plt.scatter(X_test['x'], y_test['v2'], color='blue', label='Datos reales')
plt.scatter(X_test['x'], y_test_pred[:, 1], color='red', label='Predicciones')
plt.xlabel('x')
plt.ylabel('v2')
plt.title('Predicciones vs Datos reales - Salida v2')
plt.legend()
plt.show()

residuos_v1 = y_test['v1'] - y_test_pred[:, 0]
residuos_v2 = y_test['v2'] - y_test_pred[:, 1]

plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuos_v1)), residuos_v1, color='red', label='Residuos v1')
plt.scatter(range(len(residuos_v2)), residuos_v2, color='purple', label='Residuos v2')
plt.xlabel('Índice de la muestra')
plt.ylabel('Residuos')
plt.title('Gráfico de los residuos')
plt.axhline(y=0, color='black', linestyle='--')  # Línea horizontal en y=0
plt.legend()
plt.show()


# Gráfico Q-Q de los residuos
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
stats.probplot(residuos_v1, dist="norm", plot=plt)
plt.title('Q-Q plot de los residuos para v1')
plt.xlabel('Cuantiles teóricos')
plt.ylabel('Cuantiles de los residuos')

plt.subplot(1, 2, 2)
stats.probplot(residuos_v2, dist="norm", plot=plt)
plt.title('Q-Q plot de los residuos para v2')
plt.xlabel('Cuantiles teóricos')
plt.ylabel('Cuantiles de los residuos')
plt.tight_layout()
plt.show()

# Calcular las distancias en la gráfica Q-Q para v1
residuos_sorted_v1 = np.sort(residuos_v1)
cuantiles_normales_v1 = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuos_sorted_v1)))
distancias_v1 = np.abs(residuos_sorted_v1 - cuantiles_normales_v1)
errores_ordenados_v1 = np.sort(distancias_v1)

# Calcular las distancias en la gráfica Q-Q para v2
residuos_sorted_v2 = np.sort(residuos_v2)
cuantiles_normales_v2 = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuos_sorted_v2)))
distancias_v2 = np.abs(residuos_sorted_v2 - cuantiles_normales_v2)
errores_ordenados_v2 = np.sort(distancias_v2)

# Graficar los errores en orden ascendente para v1 y v2
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(errores_ordenados_v1)), errores_ordenados_v1, marker='o', linestyle='-')
plt.title('Errores de distancia en orden ascendente para v1')
plt.xlabel('Puntos ordenados')
plt.ylabel('Error de distancia')

plt.subplot(1, 2, 2)
plt.plot(np.arange(len(errores_ordenados_v2)), errores_ordenados_v2, marker='o', linestyle='-')
plt.title('Errores de distancia en orden ascendente para v2')
plt.xlabel('Puntos ordenados')
plt.ylabel('Error de distancia')

plt.tight_layout()
plt.show()

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(tree_reg, xx, yy, cv=5, scoring='neg_mean_squared_error')
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Negative MSE')
plt.title('Curva de Aprendizaje')
plt.legend(loc='best')
plt.show()


feature_importance = tree_reg.feature_importances_
feature_names = ['x', 'y']

plt.figure(figsize=(8, 5))
plt.barh(feature_names, feature_importance, color='skyblue')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.title('Importancia de las características')
plt.show()

# Solo si tenemos dos características
if xx.shape[1] == 2:
    x_min, x_max = xx.iloc[:, 0].min() - 1, xx.iloc[:, 0].max() + 1
    y_min, y_max = xx.iloc[:, 1].min() - 1, xx.iloc[:, 1].max() + 1
    xx1, yy1 = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Superficie de decisión para v1
    Z_v1 = tree_reg.predict(np.c_[xx1.ravel(), yy1.ravel()])[:, 0]
    Z_v1 = Z_v1.reshape(xx1.shape)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.contourf(xx1, yy1, Z_v1, alpha=0.8)
    plt.scatter(xx.iloc[:, 0], xx.iloc[:, 1], c=yy['v1'], edgecolors='k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Superficie de Decisión v1')
    

    # Superficie de decisión para v2
    Z_v2 = tree_reg.predict(np.c_[xx1.ravel(), yy1.ravel()])[:, 1]
    Z_v2 = Z_v2.reshape(xx1.shape)
    plt.subplot(1, 2, 2)
    plt.contourf(xx1, yy1, Z_v2, alpha=0.8)
    plt.scatter(xx.iloc[:, 0], xx.iloc[:, 1], c=yy['v2'], edgecolors='k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Superficie de Decisión v2')
    plt.tight_layout()
    plt.show()
else:
    print("No se pueden graficar la superficie de decisión para más de dos características.")
