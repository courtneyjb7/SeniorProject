# Movement Following Animatronic Parrot
## Senior Project 2023

## Requirements
```
pip install opencv-python mediapipe pyfirmata
```
Note: Mediapipe only works with Python 3.7-3.10

## Setting Up Arduino
*   Open the Arduino application on your computer and 
click on these tabs:
    *   AdruinoIDE -> File -> Examples -> Firmata -> StandardFirmata
*   Plug in your Arduino
*   Upload the StandardFirmata code

## Run the Python Code
*   Check in your computer's Device Manager under Ports and determine what port your Arduino is plugged into.
*   Change the ```comport``` variable in [follow_controller.py](follow_controller.py) to that port
*   Run ```py follow_main.py```

