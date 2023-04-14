This repository contains Python code files for a multiprocessing-based project using Pygame and ArduinoSerial. The code is designed to work with an Arduino microcontroller and includes separate processes for managing different windows.

main.py
Main entry point for the project. It initializes and starts multiple processes for managing different windows and communicates with the Arduino microcontroller using the ArduinoSerial module.

All_class.py
This file contains the OurWindow and Animation classes, which are used to manage the windows and their respective animations. The OurWindow class manages the window creation and main loop, while the Animation class handles the updates and drawing of sprites on the screen.

ArduinoSerial.py
This module contains the ArduinoDetector class, which is responsible for communicating with the Arduino microcontroller, reading data from a specified serial port, and updating the shared Flag variable based on the input received. The module also writes input values and timestamps to a specified file.

Images.py (if applicable)
This file (not included in the provided code) should contain the Images class with the same structure as the All_class. The Images class would work specifically with images and include a flip parameter to control image orientation.

Please ensure that you have the required dependencies installed before running the project, such as Pygame, Pyserial, and win32api.
