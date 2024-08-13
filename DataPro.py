import pandas as pd
import os

def DataPro():
    # Ruta del archivo CSV
    file_path = r'C:\Users\mlarc\Desktop\RoboDK\Samples\Base de datos\Base de datos-csv'

    # Leer el archivo CSV en un DataFrame
    file_nametra = 'DataBaseTrayectoria.csv'
    csv_file_path_tra = os.path.join(file_path, file_nametra)
    rdata = pd.read_csv(csv_file_path_tra)

    # Verificar que las columnas 'v1' y 'v2' existen en el DataFrame
    if 'v1' in rdata.columns and 'v2' in rdata.columns:
        # Eliminar el primer valor de 'v1' y 'v2' y desplazar los datos hacia arriba
        rdata['v1'] = rdata['v1'].shift(-1)
        rdata['v2'] = rdata['v2'].shift(-1)

        # Eliminar la última fila que ahora contiene NaN después del desplazamiento
        rdata = rdata[:-1]

        # Guardar el DataFrame actualizado en un nuevo archivo CSV (o sobrescribir el existente)
        rdata.to_csv(csv_file_path_tra, index=False)
    else:
        print("El archivo CSV no contiene las columnas 'v1' y 'v2'.")
    eceros()


def eceros():
    try:
        # Ruta del archivo CSV
        file_path = r'C:\Users\mlarc\Desktop\RoboDK\Samples\Base de datos\Base de datos-csv'
        # Leer el archivo CSV en un DataFrame
        file_nametra = 'DataBaseTrayectoria.csv'
        csv_file_path_tra = os.path.join(file_path, file_nametra)
        rdata = pd.read_csv(csv_file_path_tra)

        # Verificar que las columnas 'v1' y 'v2' existen en el DataFrame
        if 'v1' in rdata.columns and 'v2' in rdata.columns:
            # Eliminar las filas donde 'v1' o 'v2' son igual a 0
            rdata = rdata[(rdata['v1'] != 0) & (rdata['v2'] != 0)]

            # Guardar el DataFrame actualizado en un nuevo archivo CSV (o sobrescribir el existente)
            rdata.to_csv(csv_file_path_tra, index=False)

        else:
            print("El archivo CSV no contiene las columnas 'v1' y 'v2'.")
    except PermissionError as e:
        print(f"PermissionError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")



# Mostrar el DataFrame resultante
#print(df)

DataPro()
