import time
from multiprocessing import Queue
import serial
import datetime
import subprocess
import re
from time import sleep
from threading import Thread
from collections import deque

'''Welcome to the Arduino module. In this module, Flag = 0 represents no input, and Flag = 1 represents some input.
Here, we activate the movement detection, and we can change (only the threshold in the current set-up) the Arduino code.
Threads and queues are used to allow file recording without causing pickle errors.'''


class ArduinoDetector:
    # Initialize the ArduinoDetector object
    def __init__(self, Flag, file_path, port="COM5", baudrate=115200, time_flag=0.1, flag_duration=0.5):

        self.Flag = Flag
        self.file_path = file_path
        self.recording = False
        self.write_queue = Queue()
        self.port = port
        self.baudrate = baudrate
        self.time_flag = time_flag
        self.flag_duration = flag_duration

    def is_moving(self, Flag):
        # Check if the Arduino is receiving movement data
        flag_permission_change = datetime.datetime.now()
        with serial.Serial(self.port, self.baudrate, timeout=0) as ser:
            start_time = datetime.datetime.now()
            while 1:
                timestamp = round((datetime.datetime.now() - start_time).total_seconds(), 5)
                raw = ser.readline().decode('utf8').strip()
                print(raw)
                if datetime.datetime.now() >= flag_permission_change:
                    flag_permission_change = datetime.datetime.now() + datetime.timedelta(seconds=self.flag_duration)

                    if raw == '' or raw == '0' or raw == '0,0':
                        self.Flag.value = 0
                        self.write_to_queue(timestamp, "0,0")
                    else:
                        self.Flag.value = 1
                        self.write_to_queue(timestamp, raw)

                time.sleep(self.time_flag)

    def write_to_queue(self, timestamp, data):
        # Write timestamp and data to the write_queue for recording
        if self.recording:
            self.write_queue.put((timestamp, data))

    def write_to_file(self):
        # Write data from the write_queue to the specified file
        with open(self.file_path, "a") as f:
            while True:
                timestamp, data = self.write_queue.get()
                if timestamp is None and data is None:
                    break
                f.write(f"{timestamp}: {data}\n")

    def start(self, recording=False):
        # Start the ArduinoDetector with optional recording
        self.recording = recording
        t = Thread(target=self.is_moving, args=(self.Flag,))
        t.start()
        if self.recording:
            t2 = Thread(target=self.write_to_file)
            t2.start()

# ---------------------------------------------------------------------------


def update_threshold_value(file_path, new_threshold):
    # Update the threshold value in the Arduino code file
    with open(file_path, 'r') as file:
        code = file.read()

    code = re.sub(r'(int threshold = )\d+;', f'\g<1>{new_threshold};', code)

    with open(file_path, 'w') as file:
        file.write(code)


def upload_to_arduino(sketch_path, arduino_port, board_type):
    # Compile and upload the Arduino code to the Arduino board
    arduino_cli_path = r"D:\AmirA21\Desktop\arduino-cli.exe"
    compile_command = fr'"{arduino_cli_path}" compile --fqbn {board_type} "{sketch_path}"'
    upload_command = fr'"{arduino_cli_path}" upload -p {arduino_port} --fqbn {board_type} "{sketch_path}"'

    try:
        subprocess.run(compile_command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(upload_command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Arduino code uploaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.decode('utf-8')}")


def change_and_upload_threshold(file_path, new_threshold, arduino_port="COM5", board_type="arduino:avr:uno",
                                change_threshold=False):
    # Check if the threshold need to be modified
    if change_threshold:
        update_threshold_value(file_path, new_threshold)
        upload_to_arduino(fr"{file_path}", arduino_port, board_type)
    else:
        print("No action performed. Threshold value not changed.")
