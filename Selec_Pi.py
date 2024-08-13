###Tercera Lección (o Capítulo): Generación de Data Set y Aprendizaje Supervisado
#Selección de nuevo Pi (menor ddistancia con objetivo)


from robodk import *      # RoboDK API
from robolink import *    # Robot toolbox
import random
import os
import csv
import pandas as pd
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

RDK = robolink.Robolink()

#Buscar al robot en el simulador
robot = RDK.Item("Dobot Magician", ITEM_TYPE_ROBOT)
origen = RDK.Item("Origen", ITEM_TYPE_FRAME)


def dist_min():
    file_path = r'C:\Users\mlarc\Desktop\RoboDK\Samples\Base de datos\Base de datos-csv'
    file_namer = 'DataBaseRamas.csv'
    csv_file_path_r = os.path.join(file_path, file_namer)

    rdata = pd.read_csv(csv_file_path_r)

    c1 = rdata['c1']
    c2 = rdata['c2']
    c3 = 25
    c4 = 0
    pres_obj = rdata['object']
    x = rdata['x']
    y = rdata['y']
    dist = rdata['h']
    v1 = rdata['v1']
    v2 = rdata['v2']

    fin = 0
    rin = 0


    minimo = sys.float_info.max  # Inicializamos con el máximo valor posible en punto flotante
    pos = None

    file_nametra = 'DataBaseTrayectoria.csv'
    csv_file_path_tra = os.path.join(file_path, file_nametra)
    with open(csv_file_path_tra, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

    file_namemin = 'DataBaseMinimos.csv'
    csv_file_path_min = os.path.join(file_path, file_namemin)
    with open(csv_file_path_min, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            

    file_name_tot = 'DataBaseTOTAL.csv'
    csv_file_path_tot = os.path.join(file_path, file_name_tot)
    with open(csv_file_path_tot, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['c1', 'c2', 'object', 'h', 'x','y', 'v1', 'v2'])

            
            
    for i, valor in enumerate(dist):
        with open(csv_file_path_tot, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([c1[i], c2[i], pres_obj[i], dist[i], x[i], y[i], v1[i], v2[i]])
        if valor > 0 and valor < minimo:
            minimo = valor
            pos = i
    

    if pos is not None:
        #print(f"La posición del valor mínimo positivo: {pos}")
        print(f"La distancia mínima pos es: {dist[pos]}")
        #print(f"El valor de c1 es: {c1[pos]}")
        #print(f"El valor ede c2 es: {c2[pos]}")
        robot.setJoints([c1[pos], c2[pos], c3, c4])
        with open(csv_file_path_min, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([c1[pos], c2[pos], pres_obj[pos], dist[pos], x[pos], y[pos], v1[pos],  v2[pos]])
        if v1[pos] != 0 or v2[pos] != 0:
            with open(csv_file_path_tra, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([c1[pos], c2[pos], pres_obj[pos], dist[pos], x[pos], y[pos], v1[pos],  v2[pos]])
        elif dist[pos] <= 2:
            fin = 1
        elif v1[pos]== 0 or v2[pos] == 0:
            rin = 1
        else:
            fin = 0

    
    
    else:
        print("No se encontró ningún valor mayor que 0 en la lista.")

    return fin, rin

#dist_min()
