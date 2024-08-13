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
X = rdata[['x', 'y']]
y_v1 = rdata['v1']
y_v2 = rdata['v2']

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba para cada salida
X_train, X_test, y_train_v1, y_test_v1 = train_test_split(X, y_v1, test_size=0.2, random_state=25)
_, _, y_train_v2, y_test_v2 = train_test_split(X, y_v2, test_size=0.2, random_state=25)

# Crear el modelo de árbol de decisión para cada salida
tree_reg_v1 = DecisionTreeRegressor(random_state=50)
tree_reg_v2 = DecisionTreeRegressor(random_state=50)

# Entrenar los modelos
tree_reg_v1.fit(X_train, y_train_v1)
tree_reg_v2.fit(X_train, y_train_v2)

# Predicciones en el conjunto de prueba
y_test_pred_v1 = tree_reg_v1.predict(X_test)
y_test_pred_v2 = tree_reg_v2.predict(X_test)

# Calcular el error cuadrático medio para cada salida
mse_test_v1 = mean_squared_error(y_test_v1, y_test_pred_v1)
mse_test_v2 = mean_squared_error(y_test_v2, y_test_pred_v2)
print("Error cuadrático medio en conjunto de prueba para v1:", mse_test_v1)
print("Error cuadrático medio en conjunto de prueba para v2:", mse_test_v2)



from sklearn.metrics import r2_score

r2v1 = r2_score(y_test_v1, y_test_pred_v1)
print("Coeficiente de Determinación (R²):", r2v1)
r2v2 = r2_score(y_test_v2, y_test_pred_v2)
print("Coeficiente de Determinación (R²):", r2v2)


plt.figure(figsize=(10, 6))
plt.scatter(X['x'], X['y'], color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Diagrama de dispersión de las entradas')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_v1)), y_v1, color='green', label='v1')
plt.scatter(range(len(y_v2)), y_v2, color='orange', label='v2')
plt.xlabel('Índice de la muestra')
plt.ylabel('Valores de salida')
plt.title('Diagrama de dispersión de las salidas')
plt.legend()
plt.show()

# Graficar predicciones para la salida v1
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_test['x'], y_test_v1, color='blue', label='Datos reales')
plt.scatter(X_test['x'], y_test_pred_v1, color='red', label='Predicciones')
plt.xlabel('x')
plt.ylabel('v1')
plt.title('Predicciones vs Datos reales - Salida v1')
plt.legend()


# Graficar predicciones para la salida v2
plt.subplot(1, 2, 2)
plt.scatter(X_test['x'], y_test_v2, color='blue', label='Datos reales')
plt.scatter(X_test['x'], y_test_pred_v2, color='red', label='Predicciones')
plt.xlabel('x')
plt.ylabel('v2')
plt.title('Predicciones vs Datos reales - Salida v2')
plt.legend()
plt.show()

residuos_v1 = y_test_v1 - y_test_pred_v1
residuos_v2 = y_test_v2 - y_test_pred_v2

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

# Calcular los residuos
residuos_v1 = y_test_v1 - y_test_pred_v1
residuos_v2 = y_test_v2 - y_test_pred_v2

# Calcular los cuantiles teóricos de la distribución normal
residuos_sorted_v1 = np.sort(residuos_v1)
residuos_sorted_v2 = np.sort(residuos_v2)
cuantiles_normales_v1 = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuos_sorted_v1)))
cuantiles_normales_v2 = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuos_sorted_v2)))

# Calcular las distancias entre los residuos y la línea teórica
distancias_v1 = np.abs(residuos_sorted_v1 - cuantiles_normales_v1)
distancias_v2 = np.abs(residuos_sorted_v2 - cuantiles_normales_v2)

# Graficar los errores de cada punto en la gráfica Q-Q
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(distancias_v1)), distancias_v1, marker='o', linestyle='-')
plt.title('Errores de distancia en orden ascendente para v1')
plt.xlabel('Puntos ordenados')
plt.ylabel('Error de distancia')

plt.subplot(1, 2, 2)
plt.plot(np.arange(len(distancias_v2)), distancias_v2, marker='o', linestyle='-')
plt.title('Errores de distancia en orden ascendente para v2')
plt.xlabel('Puntos ordenados')
plt.ylabel('Error de distancia')

plt.tight_layout()
plt.show()


from sklearn.model_selection import learning_curve

train_sizes, train_scores_v1, test_scores_v1 = learning_curve(tree_reg_v1, X, y_v1, cv=5, scoring='neg_mean_squared_error')
train_scores_mean_v1 = -np.mean(train_scores_v1, axis=1)
test_scores_mean_v1 = -np.mean(test_scores_v1, axis=1)

plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
plt.plot(train_sizes, train_scores_mean_v1, 'o-', color='blue', label='Training score v1')
plt.plot(train_sizes, test_scores_mean_v1, 'o-', color='green', label='Cross-validation score v1')
plt.xlabel('Training examples')
plt.ylabel('Negative MSE')
plt.title('Curva de Aprendizaje v1')
plt.legend(loc='best')


train_sizes, train_scores_v2, test_scores_v2 = learning_curve(tree_reg_v2, X, y_v2, cv=5, scoring='neg_mean_squared_error')
train_scores_mean_v2 = -np.mean(train_scores_v2, axis=1)
test_scores_mean_v2 = -np.mean(test_scores_v2, axis=1)

plt.subplot(1, 2, 2)
plt.plot(train_sizes, train_scores_mean_v2, 'o-', color='blue', label='Training score v2')
plt.plot(train_sizes, test_scores_mean_v2, 'o-', color='green', label='Cross-validation score v2')
plt.xlabel('Training examples')
plt.ylabel('Negative MSE')
plt.title('Curva de Aprendizaje v2')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

# Solo si tenemos dos características
if X.shape[1] == 2:
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Superficie de decisión para v1
    Z_v1 = tree_reg_v1.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_v1 = Z_v1.reshape(xx.shape)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z_v1, alpha=0.8)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_v1, edgecolors='k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Superficie de Decisión v1')
    

    # Superficie de decisión para v2
    Z_v2 = tree_reg_v2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_v2 = Z_v2.reshape(xx.shape)
    plt.subplot(1, 2, 2)
    plt.contourf(xx, yy, Z_v2, alpha=0.8)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_v2, edgecolors='k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Superficie de Decisión v2')
    plt.tight_layout()
    plt.show()
else:
    print("No se pueden graficar la superficie de decisión para más de dos características.")
