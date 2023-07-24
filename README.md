# Movement Following Animatronic Parrot 

Courtney Barber - Senior Project 2023

Cal Poly San Luis Obispo

<img src="https://github.com/courtneyjb7/robot_parrot/blob/main/Images/parrot.png" width="300"><img src="https://github.com/courtneyjb7/robot_parrot/blob/main/Images/arm-up.jpg" width="300">

<details open="open">
<summary>Table of Contents</summary>
<br>

- [Requirements](#Requirements)
- [Setting Up Arduino](#Setting-Up-Arduino)
- [Run the Python Code](#Run-the-Python-Code)
- [Aknowledgements](#Aknowledgements)
</details>

## Requirements
```
pip install opencv-python mediapipe pyfirmata
```
Note: Mediapipe only works with Python 3.7-3.10


## Setting Up Arduino
This part only needs to be done once.
*   Open the Arduino application on your computer and 
click on these tabs:
    *   AdruinoIDE -> File -> Examples -> Firmata -> StandardFirmata
*   Plug in your Arduino
*   Upload the StandardFirmata code


## Run the Python Code
*   Check in your computer's Device Manager under Ports and determine what port your Arduino is plugged into.
*   Change the ```comport``` variable in [follow_controller.py](follow_controller.py) to that port
*   Run ```py follow_main.py```


## Aknowledgements
See the [final report](report.pdf) for all formatted references.
*   Senior Project Advisor: Dr. John Seng
*   Education on MediaPipe set up came from:
    *   [Mediapipe documentation](https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md) 
    *   [YouTube tutorial - AI Pose Estimation with Python and MediaPipe | Plus AI Gym Tracker Project](https://www.youtube.com/watch?v=06TE_U21FK4).
*   Head detection
 	*   [Youtube tutorial - Real-Time Head Pose Estimation: A Python Tutorial with MediaPipe and OpenCV](https://www.youtube.com/watch?v=-toNMaS4SeQ&t=747s)
        *   With [this GitHub code](https://github.com/niconielsen32/ComputerVision/blob/master/headPoseEstimation.py)
*   Education on accessing Arduino from python code came from:
    *   [YouTube tutorial - How to controll LED Using Python,Mediapipe & Arduino](https://www.youtube.com/watch?v=fwMjVZhM08s&t=307s)

