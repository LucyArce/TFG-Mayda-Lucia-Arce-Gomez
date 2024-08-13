from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import csv

# Datos de entrada y salida (ejemplo)
file_path = r'E:\RoboDK\Samples\Base de datos\DataSet1'
file_namemin = 'DataBaseMinimos.csv'
csv_file_path_min = os.path.join(file_path, file_namemin)

rdata = pd.read_csv(csv_file_path_min)

# Seleccionar las características (x, y) y las salidas (v1, v2)
X = rdata[['x', 'y']]
y = rdata[['v1', 'v2']]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

# Crear y entrenar el modelo MLP
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000)
model.fit(X_train, y_train)

# Hacer predicciones
predictions = model.predict(X_test)

# Calcular el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, predictions)
print("Error cuadrático medio:", mse)

# Calcular el coeficiente de determinación (R²)
r2 = r2_score(y_test, predictions, multioutput='variance_weighted')
print("Coeficiente de determinación (R²):", r2)

# Visualizar las predicciones vs. los valores reales
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test['v1'], predictions[:, 0])
plt.xlabel('Valores reales (v1)')
plt.ylabel('Predicciones (v1)')
plt.title('Predicciones vs. Valores reales (v1)')

plt.subplot(1, 2, 2)
plt.scatter(y_test['v2'], predictions[:, 1])
plt.xlabel('Valores reales (v2)')
plt.ylabel('Predicciones (v2)')
plt.title('Predicciones vs. Valores reales (v2)')

plt.tight_layout()
plt.show()

# Gráfico de residuos
residuos_v1 = y_test['v1'] - predictions[:, 0]
residuos_v2 = y_test['v2'] - predictions[:, 1]
plt.figure(figsize=(10, 6))

plt.scatter(range(len(residuos_v1)), residuos_v1, color='red', label='Residuos v1')
plt.scatter(range(len(residuos_v2)), residuos_v2, color='purple', label='Residuos v2')
plt.xlabel('Índice de la muestra')
plt.ylabel('Residuos')
plt.title('Gráfico de los residuos')
plt.axhline(y=0, color='black', linestyle='--')  # Línea horizontal en y=0
plt.legend()
plt.show()


# Calcular los residuos
residuos_v1 = y_test['v1'] - predictions[:, 0]
residuos_v2 = y_test['v2'] - predictions[:, 1]

# Gráfico Q-Q de los residuos
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
stats.probplot(residuos_v1, dist="norm", plot=plt)
plt.title('Q-Q plot de los residuos para v1')

plt.subplot(1, 2, 2)
stats.probplot(residuos_v2, dist="norm", plot=plt)
plt.title('Q-Q plot de los residuos para v2')

plt.tight_layout()
plt.show()

# Errores de Q-Q en orden ascendente
sorted_residuos_v1 = np.sort(residuos_v1)
sorted_residuos_v2 = np.sort(residuos_v2)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(sorted_residuos_v1, 'o', color='blue')
plt.title('Errores de Q-Q en orden ascendente para v1')
plt.xlabel('Índice')
plt.ylabel('Residuos ordenados')

plt.subplot(1, 2, 2)
plt.plot(sorted_residuos_v2, 'o', color='green')
plt.title('Errores de Q-Q en orden ascendente para v2')
plt.xlabel('Índice')
plt.ylabel('Residuos ordenados')

plt.tight_layout()
plt.show()

from sklearn.model_selection import learning_curve

# Generar las curvas de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Promedio de las puntuaciones
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

# Graficar la curva de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Puntuación de entrenamiento')
plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Puntuación de validación')
plt.xlabel('Número de ejemplos de entrenamiento')
plt.ylabel('MSE negativo')
plt.title('Curva de aprendizaje')
plt.legend(loc='best')
plt.show()


# Generar una malla de puntos en el espacio de características
x_min, x_max = X['x'].min() - 1, X['x'].max() + 1
y_min, y_max = X['y'].min() - 1, X['y'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predecir valores para cada punto en la malla
Z_v1 = model.predict(np.c_[xx.ravel(), yy.ravel()])[:, 0].reshape(xx.shape)
Z_v2 = model.predict(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

# Graficar la superficie de decisión
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_v1, alpha=0.8)
plt.scatter(X['x'], X['y'], c=y['v1'], edgecolor='k', cmap=plt.cm.coolwarm)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Superficie de Decisión para v1')

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_v2, alpha=0.8)
plt.scatter(X['x'], X['y'], c=y['v2'], edgecolor='k', cmap=plt.cm.coolwarm)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Superficie de Decisión para v2')

plt.tight_layout()
plt.show()
