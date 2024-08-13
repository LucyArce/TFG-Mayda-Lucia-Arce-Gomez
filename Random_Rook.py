# Forward and backwards compatible use of the RoboDK API:
# Remove these 2 lines to follow python programming guidelines
from robodk import *      # RoboDK API
from robolink import *    # Robot toolbox
import random
RDK = robolink.Robolink()

rook = RDK.Item("Rook", ITEM_TYPE_OBJECT)


def rrook():
    rx = random.randint(-121, -57)
    ry = random.randint(-104, 104)
    
    rook.setPose(transl(rx,ry,0))


#rrook()