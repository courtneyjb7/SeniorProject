# Help from https://www.youtube.com/watch?v=fwMjVZhM08s&t=307s

import pyfirmata

# check in Device Manager under Ports
comport='COM3'

board = pyfirmata.Arduino(comport)
high = 135
mid = 90
low = 45

leftA_pin3 = board.get_pin("d:3:s")
rightA_pin5 = board.get_pin("d:5:s")
head_pin11 = board.get_pin("d:11:s")

# The angle for the arms are flipped
def angleSwitchSide(angle):
    return high - angle + low

### Parrot Mirrors Person ###

def leftArm(angle):
    leftA_pin3.write(angle)

def rightArm(angle):
    rightA_pin5.write(angleSwitchSide(angle))

def head(angle):
    head_pin11.write(angleSwitchSide(angle))


### Parrot match person instead of mirror ###
# def leftArm(angle):
#     rightA_pin5.write(angleSwitchSide(angle))

# def rightArm(angle):
#     leftA_pin3.write(angle)

# def head(angle):
#     head_pin11.write(angle)
