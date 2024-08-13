#Llamado de cámara, captura de imagen y preprocesado de imagen




from robodk import *      # RoboDK API
from robolink import *    # Robot toolbox
import random
import os
import csv
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

RDK = robolink.Robolink()
RDK.Cam2D_Close()

# Buscar el marco de referencia de la cámara y agregar la cámara
camref = RDK.Item('CAMREF', ITEM_TYPE_FRAME)
cam_id = RDK.Cam2D_Add(camref, 'FOCAL_LENGHT=4.8 FOV=51 FAR_LENGHT=290 SIZE=320x240 BG_COLOR=white')

# Buscar al robot en el simulador
robot = RDK.Item("Dobot Magician", ITEM_TYPE_ROBOT)

# Definir rangos de colores en HSV
lower_range1 = (0, 100, 20)  # Rojo
upper_range1 = (10, 255, 255)

lower_range2 = (12, 25, 25)  # Verde
upper_range2 = (86, 255, 255)

lower_range3 = (78, 158, 125)  # Azul
upper_range3 = (136, 255, 255)


def rcam(i):
    file_path = r'C:\Users\mlarc\Desktop\RoboDK\Samples\Base de datos'
    # Tomar foto y leer la foto
    RDK.Cam2D_Snapshot(file_path + "/cam.png", cam_id)
    img = cv.imread(file_path + "/cam.png")

    # Convertir imagen a HSV
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Máscaras identificadoras de colores
    mask1 = cv.inRange(hsv_img, lower_range1, upper_range1)
    color_image1 = cv.bitwise_and(img, img, mask=mask1)

    mask2 = cv.inRange(hsv_img, lower_range2, upper_range2)
    color_image2 = cv.bitwise_and(img, img, mask=mask2)

    mask3 = cv.inRange(hsv_img, lower_range3, upper_range3)
    color_image3 = cv.bitwise_and(img, img, mask=mask3)

    blue = color_image3
    red = color_image1
    green = color_image2

    # Selección del color
    colour = green

    # Binarización de imagen
    img1 = cv.cvtColor(colour, cv.COLOR_HSV2BGR)
    img1 = cv.cvtColor(colour, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img1, 0, 256, cv.THRESH_BINARY)
    Ima2 = thresh.copy()

    # Guardar imagen con el dato correspondiente
    if not os.path.exists(file_path):
        print(f"Error: La ruta {file_path} no existe.")
        return None, None
    file_name = "Data" + str(i) + ".png"
    cv.imwrite(os.path.join(file_path , file_name), Ima2)
    

    # Encontrar píxeles blancos (coordenadas y # tot)
    white_pixels = np.column_stack(np.where(Ima2 == 255))
    white_pix_tot = np.sum(Ima2 == 255)

    if 0 < white_pix_tot:
    # Calcular la media de las coordenadas de los píxeles blancos
        mean_coordinates = white_pixels.mean(axis=0)
        x, y = mean_coordinates

    else:
        x = 500
        y = 500
    
    #print('X es:', x)
    #print('Y es:', y)
    return x,y

# Llamar a la función
#x, y = rcam(1)

