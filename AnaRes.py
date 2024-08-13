import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os

# Cargar los datos desde el archivo Excel
file_path = r'C:\Users\mlarc\Desktop\RoboDK\Samples\Base de datos'
file_name_res = 'ResultadosRED.csv'
csv_file_path_res = os.path.join(file_path, file_name_res)
rdata = pd.read_csv(csv_file_path_res)


# Crear el scatter plot para v1r vs v1p
plt.figure(figsize=(10, 6))
plt.scatter(rdata['Valor Real v1'], rdata['Valor Predicho v1'], alpha=0.5)
plt.xlabel('Valor Real v1')
plt.ylabel('Valor Predicho v1')
plt.title('Valor Real v1 vs Valor Predicho v1')
plt.grid(True)
plt.show()


# Crear el scatter plot para v1r vs v1p
plt.figure(figsize=(10, 6))
plt.scatter(rdata['Valor Real v2'], rdata['Valor Predicho v2'], alpha=0.5)
plt.xlabel('Valor Real v2')

plt.ylabel('Valor Predicho v2')
plt.title('Valor Real v2 vs Valor Predicho v2')
plt.grid(True)
plt.show()


# Obtener los valores únicos de X y Y
X_values = rdata['X'].unique()
Y_values = rdata['Y'].unique()

# Generar todas las combinaciones posibles de X e Y
combinations = list(itertools.product(X_values, Y_values))

# Crear un DataFrame con todas las combinaciones
combinations_rdata = pd.DataFrame(combinations, columns=['X', 'Y'])

# Mezclar las combinaciones con los valores de v1p
merged_rdata = pd.merge(combinations_rdata, rdata[['X', 'Y', 'Valor Predicho v1']], on=['X', 'Y'], how='left')

# Crear el scatter plot para v1p
plt.figure(figsize=(10, 6))
plt.scatter(merged_rdata['X'], merged_rdata['Y'], c=merged_rdata['Valor Predicho v1'], cmap='viridis')
plt.colorbar(label='Valor Predicho v1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot de todas las combinaciones posibles de X e Y para Valor Predicho v1')
plt.show()

# Mezclar las combinaciones con los valores de v2p
merged_rdata = pd.merge(combinations_rdata, rdata[['X', 'Y', 'Valor Predicho v2']], on=['X', 'Y'], how='left')

# Crear el scatter plot para v2p
plt.figure(figsize=(10, 6))
plt.scatter(merged_rdata['X'], merged_rdata['Y'], c=merged_rdata['Valor Predicho v2'], cmap='viridis')
plt.colorbar(label='Valor Predicho v2')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot de todas las combinaciones posibles de X e Y para Valor Predicho v2')
plt.show()


import pandas as pd
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Cargar los datos desde el archivo Excel

# Obtener los valores únicos de X y Y
X_values = rdata['X'].unique()
Y_values = rdata['Y'].unique()

# Generar todas las combinaciones posibles de X e Y
combinations = list(itertools.product(X_values, Y_values))

# Crear un DataFrame con todas las combinaciones
combinations_rdata = pd.DataFrame(combinations, columns=['X', 'Y'])

# Mezclar las combinaciones con los valores de v1p
merged_rdata = pd.merge(combinations_rdata, rdata[['X', 'Y', 'Valor Predicho v1']], on=['X', 'Y'], how='left')

# Crear el scatter plot 3D para v1p
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(merged_rdata['X'], merged_rdata['Y'], merged_rdata['Valor Predicho v1'], c=merged_rdata['Valor Predicho v1'], cmap='viridis')

# Añadir etiquetas y título
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('v1p')
ax.set_title('Scatter Plot 3D de todas las combinaciones posibles de X e Y para Valor Predicho v1')

# Añadir barra de color
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Valor Predicho v1')

# Mostrar el plot
plt.show()



# Crear el scatter plot 3D para v2p

sc2 = ax.scatter(merged_rdata['X'], merged_rdata['Y'], merged_rdata['Valor Predicho v2'], c=merged_rdata['Valor Predicho v2'], cmap='viridis')

# Añadir etiquetas y título
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('v2p')
ax.set_title('Scatter Plot 3D de todas las combinaciones posibles de X e Y para Valor Predicho v2')

# Añadir barra de color
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Valor Predicho v2')

# Mostrar el plot
plt.show()

