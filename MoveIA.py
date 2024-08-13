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
import IA_entrena

RDK = robolink.Robolink()

# Buscar al robot en el simulador
robot = RDK.Item("Dobot Magician", ITEM_TYPE_ROBOT)

def MoveIA(Mode):
    xp, yp = rcam.rcam(0)

    xo = 120
    yo = 205
    error_x = xo - xp
    error_y = yo - yp
    x2 = error_x ** 2
    y2 = error_y ** 2
    h2 = x2 + y2
    h = sqrt(h2)
    #print (h)
    while h >= 2:
        xp, yp = rcam.rcam(0)
        v1, v2 = IA_entrena.modelo_usar(xp, yp, Mode)
        joints = robot.Joints().list()
        c1, c2, c3, c4 = joints
        c1r = c1 + v1
        c2r = c2 + v2
        robot.setJoints([c1r, c2r, c3, c4])

    
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
        #print (xp)
        #print (yp)
        #print (h)
    PickApproachPosition = robot.Pose() * transl(0, 0, 40)
    robot.MoveJ(PickApproachPosition)

# Llamar a la función
while True:
    MoveIA(2)
    # Obtener la posición actual del robot
    pose = robot.Pose()
    z = pose.Pos()[2]
    #print(f"Altura actual de z: {z}")
    if z <= 65.0:
        PickUp = robot.Pose() * transl(-10, 0, 50)
        robot.MoveJ(PickUp)
        break
