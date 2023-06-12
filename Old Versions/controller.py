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
def angleSwitchArm(angle):
    return high - angle + low

def leftArm(position):
    if position=="up":
        angle = high
    else:
        angle = low
    leftA_pin3.write(angle)

def rightArm(position):
    if position=="up":
        angle = low
    else:
        angle = high
    rightA_pin5.write(angle)

def head(position):
    if position=="left":
        angle = low
    elif position=="right":
        angle = high
    else:
        angle = mid
    head_pin11.write(angle)
