import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
import matplotlib.pyplot as plt
import os
import csv
import scipy.stats as stats


file_path = r'E:\RoboDK\Samples\Base de datos\DataSet1'
file_namemin = 'DataBaseMinimos.csv'
csv_file_path_min = os.path.join(file_path, file_namemin)

rdata = pd.read_csv(csv_file_path_min)

# Seleccionar las características (x, y) y las salidas (v1, v2)
X = rdata[['x', 'y']]
y_v1 = rdata['v1']
y_v2 = rdata['v2']


X_train, X_test, y_train_v1, y_test_v1 = train_test_split(X, y_v1, test_size=0.2, random_state=42)
_, _, y_train_v2, y_test_v2 = train_test_split(X, y_v2, test_size=0.2, random_state=42)

knn_v1 = KNeighborsRegressor(n_neighbors=10)
knn_v1.fit(X_train, y_train_v1)
y_pred_v1 = knn_v1.predict(X_test)

knn_v2 = KNeighborsRegressor(n_neighbors=10)
knn_v2.fit(X_train, y_train_v2)
y_pred_v2 = knn_v2.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

mse_test_v1 = mean_squared_error(y_test_v1, y_pred_v1)
mse_test_v2 = mean_squared_error(y_test_v2, y_pred_v2)
mae_test_v1 = mean_absolute_error(y_test_v1, y_pred_v1)
mae_test_v2 = mean_absolute_error(y_test_v2, y_pred_v2)
r2_test_v1 = r2_score(y_test_v1, y_pred_v1)
r2_test_v2 = r2_score(y_test_v2, y_pred_v2)

print("MSE v1:", mse_test_v1)
print("MAE v1:", mae_test_v1)
print("R² v1:", r2_test_v1)
print("MSE v2:", mse_test_v2)
print("MAE v2:", mae_test_v2)
print("R² v2:", r2_test_v2)



# Visualización de las predicciones vs valores reales
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test_v1, y_pred_v1, color='blue')
plt.plot([y_test_v1.min(), y_test_v1.max()], [y_test_v1.min(), y_test_v1.max()], 'k--', lw=3)
plt.xlabel('Valores Reales v1')
plt.ylabel('Valores Predichos v1')
plt.title('Predicciones vs Valores Reales para v1')

plt.subplot(1, 2, 2)
plt.scatter(y_test_v2, y_pred_v2, color='green')
plt.plot([y_test_v2.min(), y_test_v2.max()], [y_test_v2.min(), y_test_v2.max()], 'k--', lw=3)
plt.xlabel('Valores Reales v2')
plt.ylabel('Valores Predichos v2')
plt.title('Predicciones vs Valores Reales para v2')

plt.tight_layout()
plt.show()

# Gráfico de residuos
residuos_v1 = y_test_v1 - y_pred_v1
residuos_v2 = y_test_v2 - y_pred_v2

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

plt.subplot(1, 2, 2)
stats.probplot(residuos_v2, dist="norm", plot=plt)
plt.title('Q-Q plot de los residuos para v2')

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


# Calcular las curvas de aprendizaje para el modelo v1
train_sizes_v1, train_scores_v1, test_scores_v1 = learning_curve(
    knn_v1, X_train, y_train_v1, cv=5, scoring='neg_mean_squared_error'
)
train_scores_mean_v1 = -np.mean(train_scores_v1, axis=1)
test_scores_mean_v1 = -np.mean(test_scores_v1, axis=1)

# Calcular las curvas de aprendizaje para el modelo v2
train_sizes_v2, train_scores_v2, test_scores_v2 = learning_curve(
    knn_v2, X_train, y_train_v2, cv=5, scoring='neg_mean_squared_error'
)
train_scores_mean_v2 = -np.mean(train_scores_v2, axis=1)
test_scores_mean_v2 = -np.mean(test_scores_v2, axis=1)

# Graficar las curvas de aprendizaje para v1 y v2
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(train_sizes_v1, train_scores_mean_v1, 'o-', color='blue', label='Training score')
plt.plot(train_sizes_v1, test_scores_mean_v1, 'o-', color='green', label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Negative MSE')
plt.title('Curva de Aprendizaje para v1')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.plot(train_sizes_v2, train_scores_mean_v2, 'o-', color='blue', label='Training score')
plt.plot(train_sizes_v2, test_scores_mean_v2, 'o-', color='green', label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Negative MSE')
plt.title('Curva de Aprendizaje para v2')
plt.legend(loc='best')

plt.tight_layout()
plt.show()


# Visualización de la superficie de decisión

# Generar una malla de puntos en el espacio de características
x_min, x_max = X['x'].min() - 1, X['x'].max() + 1
y_min, y_max = X['y'].min() - 1, X['y'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predecir valores para cada punto en la malla
Z_v1 = knn_v1.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_v2 = knn_v2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Graficar la superficie de decisión
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_v1, alpha=0.8)
plt.scatter(X['x'], X['y'], c=y_v1, edgecolor='k', cmap=plt.cm.coolwarm)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Superficie de Decisión para v1')

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_v2, alpha=0.8)
plt.scatter(X['x'], X['y'], c=y_v2, edgecolor='k', cmap=plt.cm.coolwarm)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Superficie de Decisión para v2')

plt.tight_layout()
plt.show()