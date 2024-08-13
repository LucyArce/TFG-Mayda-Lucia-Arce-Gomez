#Ejemplo 100% programado
#Llamar a las librerías
from robolink import *
from robodk import *
#from robodk.robomath import *
RDK = Robolink()

#Crear el robot en el código
robot = RDK.Item("Dobot Magician", ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception('No se puedo encontrar el Dobot Magician')

#Crear el frame del robot
robotFrame = robot.Parent()
if robotFrame.Type() != ITEM_TYPE_FRAME:
    raise Exception('Robot parent is not a frame')


#Empieza el programa
def Picknplace():
    #creación del item para el programa
    Picknplace = RDK.Item('Picknplace', ITEM_TYPE_PROGRAM)
    if Picknplace.Valid():
        raise Exception('Programa con el nombre Picknplace ya existe.')
    Picknplace = RDK.AddProgram('Picknplace')
    Picknplace.setParam("Tree", "collapse")

#Origen
    Origen = RDK.Item("Origen", ITEM_TYPE_FRAME)
    if not Origen.Valid():
        Origen = RDK.AddFrame("Origen")
        Origen.setPose(Pose(0.000, 0.000, 0.000, -0.000, 0.000, -0.000))
        Origen.setParent('robotFrame')
    Picknplace.setPoseFrame(Origen)

#1 Movimiento de ejes, inicio
    AutoTarget0 = RDK.AddTarget('Target 0', robotFrame, robot)
    AutoTarget0.setPose(Pose(208.000, 0.000, 218.000, 180.000, 0.000, -180.000))
    Picknplace.MoveJ(AutoTarget0)
    Picknplace.WaitMove(500)
#2 Movimiento ejes, busca de objeto
    AutoTarget1 = RDK.AddTarget('Target 1', robotFrame, robot)
    AutoTarget1.setPose(Pose(317.383, 0.000, 12.607, 180.000, 0.000, -180.000))
    Picknplace.MoveJ(AutoTarget1)

#3 acerca a objeto
    Picknplace.MoveJ(AutoTarget0)

#4 acercamiento a objetivo
    AutoTarget2 = RDK.AddTarget('Target 2', robotFrame, robot)
    AutoTarget2.setPose(Pose(0.000, 208.000, 218.000, 180.000, 0.000, 90.000))
    Picknplace.MoveJ(AutoTarget2)

#5 acercamiento2 objetivo
    AutoTarget3 = RDK.AddTarget('Target 3', robotFrame, robot)
    AutoTarget3.setPose(Pose(0.000, 276.686, 25.512, 180.000, 0.000, 90.000))
    Picknplace.MoveJ(AutoTarget3)

#6 alejamiento
    Picknplace.MoveJ(AutoTarget2)

#7 inicio
    Picknplace:MoveJ(AutoTarget0)


    return


#Llamar al main
Picknplace()
