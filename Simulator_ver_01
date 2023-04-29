import pygame
from pygame.locals import *
import ArduinoSerial
import All_class
# import Images
from multiprocessing import Process, Value
from win32api import GetSystemMetrics

'''I appreciate your interest in my work, and I'm glad to provide an overview of the core concepts behind this code.
This code, which runs smoothly on Python 3.9, is based on the idea of multiprocessing and consists of 3 concurrent 
processes, excluding the main process. The goal is to create a more modular, flexible, and powerful system by using 
pygame and multiprocessing.
The Images class shares the same structure as the All_class, but specifically works with images and includes a flip 
parameter to control image orientation. Each window in this system is a separate process, making it easy to modify and
adapt the code. The sensor module, ArduinoSerial, works with an Arduino microcontroller, which has its own code to 
interact with any USB device (as described in the method).
A C code (Arduino_C_Code) enables the reception and utilization of USB input. The Arduino process modifies the value of
a shared boolean variable that represents the USB device's status (for example, in the case of a mouse sensor, True 
indicates movement detection). Each window loop checks this variable's status during each iteration and adjusts its
behavior accordingly. It's worth noting that each device can operate within its own module. For instance, Arduino input 
can be recorded by adding a thread to its process.'''

# The windows coordinates are based on the Windows screen sharing settings. It needs to be adjusted accordingly.
win_pos_1 = (1930, 200)
win_pos_2 = (3850, 200)

# The win_fb parameter controls the simulation direction, allowing for easy modification as needed.
Win_1 = All_class.OurWindow(win_pos_1, win_fb=True, state=FULLSCREEN)
Win_2 = All_class.OurWindow(win_pos_2, win_fb=False, state=FULLSCREEN)
#  import Images and Images.OurWindow add flip for images


def main():

    file_path = 'idontcare.txt'  # Capture the input values received from the Arduino.
    detector = ArduinoSerial.ArduinoDetector(Flag, file_path)
    p1 = Process(target=Win_1.run_experiment, args=(Flag,))
    p2 = Process(target=Win_2.run_experiment, args=(Flag,))
    p3 = Process(target=ArduinoSerial.ArduinoDetector.start, args=(Flag, file_path))
    p3 = Process(target=detector.start)
    p1.start()
    p2.start()
    p3.start()


if __name__ == "__main__":

    Flag = Value('i', False)
    main()
    pygame.quit()
    exit()
