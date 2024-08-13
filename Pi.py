###Tercera Lección (o Capítulo): Generación de Data Set y Aprendizaje Supervisado
#Creación de 20 posiciones cercanas a P inicial



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

import rcam

#Buscar al robot en el simulador
robot = RDK.Item("Dobot Magician", ITEM_TYPE_ROBOT)
origen = RDK.Item("Origen", ITEM_TYPE_FRAME)
home = RDK.Item("Home", ITEM_TYPE_TARGET)
gripper = RDK.Item("Gripper (Open)")
base = RDK.Item("Magician Base", ITEM_TYPE_FRAME)


def Pi():
    
    c3 = 25
    c4 = 0

    file_path = r'C:\Users\mlarc\Desktop\RoboDK\Samples\Base de datos\Base de datos-csv'

    file_nametra = 'DataBaseTrayectoria.csv'
    csv_file_path_tra = os.path.join(file_path, file_nametra)

    rdata = pd.read_csv(csv_file_path_tra)

    c1 = rdata['c1']
    c2 = rdata['c2']
    ex = rdata['x']
    ey = rdata['y']
    hh = rdata['h']

    c1i = float(c1.iloc[-1])
    c2i = float(c2.iloc[-1])
    x = ex.iloc[-1]
    y = ey.iloc[-1]
    h = hh.iloc[-1]
    v1i = 0
    v2i = 0
    

    file_namer = 'DataBaseRamas.csv'
    csv_file_path_r = os.path.join(file_path, file_namer)
    with open(csv_file_path_r, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['c1', 'c2', 'object', 'h','x','y', 'v1', 'v2'])

    

    for i in range(21):
        ran = 0
        if i == 0:
            pres_obj =  1
            with open(csv_file_path_r, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([c1i, c2i, pres_obj, h, x, y, v1i, v2i])
            #print (c1i)
            #print (c2i)
        else:
            while ran == 0:
                r1 = random.uniform(-5, 5)
                r2 = random.uniform(-5, 5)
                c1r = c1i + r1
                c2r = c2i + r2
                if -90<c1r< 90 and 0<c2r<85:
                    ran = 1
                    robot.setJoints([c1r, c2r, c3, c4])

                    xp, yp = rcam.rcam(i)

                    if xp < 320 and yp < 320:
                        x = xp
                        y = yp
                        pres_obj = 1

                        xo = 120
                        yo = 205
                        error_x = xo - x
                        error_y = yo - y
                        x2 = error_x**2
                        y2 = error_y**2
                        h2 = x2 + y2
                        h= sqrt(h2)
        
                    else:
                        pres_obj = 0
                        x = 0
                        y = 0
                        h = 0

                    with open(csv_file_path_r, 'a', newline='') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow([c1r, c2r, pres_obj, h, x, y, r1, r2])
                        
                else:
                    ran = 0
                #print('X es:', x)
                #print('Y es:', y)

        

#Pi()
