import time
import serial
import datetime
from time import sleep
from threading import Thread
'''Welcome to the Arduino module. In this module, Flag = 0 represents no input, and Flag = 1 represents some input.'''


class ArduinoDetector:

    def __init__(self, Flag, file_path):

        self.Flag = Flag
        self.file_path = file_path

    def is_moving(self, Flag):

        port = 'COM5'  # The port that communicates with the Arduino, check Windows settings
        baudrate = 115200  # Number of bits transmitted per second

        with serial.Serial(port, baudrate, timeout=0) as ser:
            start = datetime.datetime.now()
            while 1:
                with open(self.file_path, "a") as f:  # 'a' is for append mode

                    timestamp = round((datetime.datetime.now() - start).total_seconds(), 5)
                    raw = ser.readline().decode('utf8')
                    print(raw)

                    if raw == '' or raw is None:

                        self.Flag.value = 0
                        f.write(f"{timestamp}: 0\n")
                        time.sleep(0.01)  # Latency is important to avoid pipe overflow
                        continue

                    else:
                        self.Flag.value = 1
                        f.write(f'{timestamp}: {raw}\n')
                        time.sleep(0.01)

    def start(self):
        t = Thread(target=self.is_moving(self.Flag))
        t.start()  # No issues were encountered due to not closing the thread.
