#Ejemplo 100% programado
#Llamar a las librerías
from robolink import *
from robodk import *
#from robodk.robomath import *
RDK = Robolink()

#Buscar al robot en el simulador
robot = RDK.Item("Dobot Magician", ITEM_TYPE_ROBOT)
#robotFrame = robot.Parent()
rook = RDK.Item("Rook", ITEM_TYPE_OBJECT)
origen = RDK.Item("Origen", ITEM_TYPE_FRAME)
home = RDK.Item("Home", ITEM_TYPE_TARGET)
gripper = RDK.Item("Gripper (Open)")
base = RDK.Item("Magician Base", ITEM_TYPE_FRAME)
box = RDK.Item("Box", ITEM_TYPE_OBJECT)


#set the inicial axes positions
robot.setJoints([-90, 0, -10, 0])
robot.MoveJ(home)


PosHome = home.PoseAbs()

PickUpPosition = rook.PoseAbs()
#print(PickUpPosition)
#RDK.ShowMessage("Rook " + str(PickUpPosition))

XYZABG = pose_2_xyzrpw(PickUpPosition)
#RDK.ShowMessage("Rook mat" + str(XYZABG))
X, Y, Z, A, B, G = XYZABG
#RDK.ShowMessage("X" + str(X))
#RDK.ShowMessage("Y" + str(Y))

###############################################################################
if Y > X:
    PickUpPosition = PickUpPosition * transl(0, -1003, 18) * rotx(pi)
    PickUpPosition = PickUpPosition * rotz(-pi)
else:
    RDK.ShowMessage("Caso especial")


PickApproachPosition = PickUpPosition
PickApproachPosition = PickUpPosition * transl(0, 0, -50)



#print(PosHome)
#RDK.ShowMessage("Home" + str(PosHome))
#print(PickUpPosition)
#print( PickApproachPosition)
#RDK.ShowMessage("PickUp: " + str(PickUpPosition))
RDK.ShowMessage("Approach: " + str(PickApproachPosition))

robot.MoveJ(PickApproachPosition)
robot.MoveJ(PickUpPosition)

rookPU = gripper.AttachClosest()

robot.MoveL(PickApproachPosition)
robot.MoveJ(home)

###########################################################################33333


place = box.PoseAbs()
place = place * transl(0, 277, 160)
place = place * rotz(-pi/2)

#print(place)
#RDK.ShowMessage("Place" + str(place))


PlaceApproachPosition = place
PlaceApproachPosition = PlaceApproachPosition * transl(0, 0, 50)


#RDK.ShowMessage("PlaceApp: " + str(PlaceApproachPosition))

robot.MoveJ(PlaceApproachPosition)
robot.MoveJ(place)

gripper.DetachAll()
rookPU.setParentStatic(base)

robot.MoveJ(PlaceApproachPosition)
robot.MoveJ(home)

