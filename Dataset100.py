###Tercera Lección (o Capítulo): Generación de Data Set y Aprendizaje Supervisado
#Llamado a 100 veces seleccionar un nuevo Pi


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

# Close any open 2D camera views
RDK.Cam2D_Close()

#Buscar al robot en el simulador
robot = RDK.Item("Dobot Magician", ITEM_TYPE_ROBOT)
origen = RDK.Item("Origen", ITEM_TYPE_FRAME)
home = RDK.Item("Home", ITEM_TYPE_TARGET)
gripper = RDK.Item("Gripper (Open)")
base = RDK.Item("Magician Base", ITEM_TYPE_FRAME)

import Random_Rook
#import Anticiclos

import P0
import Pi
import Selec_Pi
import DataPro

file_path = r'C:\Users\mlarc\Desktop\RoboDK\Samples\Base de datos\Base de datos-csv'

file_nametra = 'DataBaseTrayectoria.csv'
csv_file_path_tra = os.path.join(file_path, file_nametra)
with open(csv_file_path_tra, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['c1', 'c2', 'object', 'h', 'x','y', 'v1', 'v2'])

file_namemin = 'DataBaseMinimos.csv'
csv_file_path_min = os.path.join(file_path, file_namemin)
with open(csv_file_path_min, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['c1', 'c2', 'object', 'h', 'x','y', 'v1', 'v2'])

file_name_tot = 'DataBaseTOTAL.csv'
csv_file_path_tot = os.path.join(file_path, file_name_tot)
with open(csv_file_path_tot, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    #csvwriter.writerow(['c1', 'c2', 'object', 'h', 'x','y', 'v1', 'v2'])


file_name0 = 'DataBase0.csv'
csv_file_path_0 = os.path.join(file_path, file_name0)
with open(csv_file_path_0, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['c1', 'c2', 'object', 'h', 'x','y', 'v1', 'v2'])

file_namer = 'DataBaseRamas.csv'
csv_file_path_r = os.path.join(file_path, file_namer)
with open(csv_file_path_r, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['c1', 'c2', 'object', 'h', 'x','y', 'v1', 'v2'])

j = 0
fin = 1
rin = 0

while j < 100:
    if fin != 0:
        Random_Rook.rrook()
        P0.P0()
        fin = 0
        j = j - 1
        print('Inicio en', j)

    else:
        Pi.Pi()
        fin, rin = Selec_Pi.dist_min()
        if rin != 0:
            j = j
            rin = 0
        else:
            j = j + 1
    print(j)
        
        
#RDK.ShowMessage("n9" + joints)
DataPro.DataPro()
