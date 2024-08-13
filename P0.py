### Tercera Lección (o Capítulo): Generación de Data Set y Aprendizaje Supervisado
#Creación de P0 random


from robodk import *      # RoboDK API
from robolink import *    # Robot toolbox
import random
import os
import csv
import numpy as np
from matplotlib import pyplot as plt

import rcam

RDK = robolink.Robolink()

# Buscar al robot en el simulador
robot = RDK.Item("Dobot Magician", ITEM_TYPE_ROBOT)

def P0():
    c3 = 25
    c4 = 0
    pres_obj = 0
    hi = 0

    # Ruta para guardar el archivo CSV
    file_path = r'C:\Users\mlarc\Desktop\RoboDK\Samples\Base de datos\Base de datos-csv'
    file_name0 = 'DataBase0.csv'
    csv_file_path_0 = os.path.join(file_path, file_name0)

    file_nametra = 'DataBaseTrayectoria.csv'
    csv_file_path_tra = os.path.join(file_path, file_nametra)

    file_namemin = 'DataBaseMinimos.csv'
    csv_file_path_min = os.path.join(file_path, file_namemin)


    while pres_obj == 0:
        c1r = random.uniform(-22, 22)
        c2r = random.uniform(0, 85)
        robot.setJoints([c1r, c2r, c3, c4])
        v1 = 0
        v2 = 0

        # Llamar la media del objeto detectado en cámara
        xp, yp = rcam.rcam(0)

        if xp < 320 and yp < 320:
            x = xp
            y = yp
            pres_obj = 1

            xo = 120
            yo = 205
            error_x = xo - x
            error_y = yo - y
            x2 = error_x ** 2
            y2 = error_y ** 2
            h2 = x2 + y2
            h = sqrt(h2)
            with open(csv_file_path_min, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([c1r, c2r, pres_obj, h, x, y, v1, v2])

            with open(csv_file_path_tra, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([c1r, c2r, pres_obj, h, x, y, v1, v2])
        else:
            pres_obj = 0
            h = 0
            x = 0
            y = 0

        # Añadir datos al archivo CSV
        with open(csv_file_path_0, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([c1r, c2r, pres_obj, h, x, y, v1, v2])
    #print('X es:', x)
    #print('Y es:', y)

# Llamar a la función
#P0()

