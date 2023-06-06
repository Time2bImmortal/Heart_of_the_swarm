''''''

P = {
    "simulation_params": {
        "direction": ["backward", "forward"],  # The direction of the simulation ('forward' or 'backward')
        "num_sprites": 60,  # The number of sprites in the simulation
        "velocity": 2,  # The velocity of the sprites
        "shift_direction": False,  # Whether the direction should shift periodically or not (True or False)
        "shift_time": 60,  # The time interval for shifting direction, if enabled
        "black_screen_duration": 120,  # The time of the black screen, you can insert it in the experiment flow
        "trial_duration": 60
    },
    "structure_params": {
        "window_position": [[1930, 200], [3850, 200]]   # The position of the windows as a list of [x, y] coordinates
    },
    "sensor_params": {
        "port": "COM5",  # The port where the arduino is connected
        "baudrate": 115200,  # The baudrate rate (bits per seconds) value for the serial communication
        "new_threshold_value": 15,  # The new threshold value for the sensor
        "Change_threshold": True,  # Whether the threshold should be changed or not (True or False)
        "arduino_code_path": "D:\\AmirA21\\Desktop\\Yossef\\Pygame_Closed_Loop_ver_02\\Arduino_code_mouse\\Arduino_code_mouse.ino",  # The file path of the Arduino code
        "control_system": 2,  # The control system(0: open-loop, 1: closed-loop synchro with the subject, 2: closed-loop opposite to the subject
        "Recording": False,  # Whether the sensor data should be recorded or not (True or False)
        "recording_file": "Mouse_records_01.txt",  # The name of the file to store the recorded sensor data
        "time_flag_duration": 0.1,
        "flag_duration": 0.7
    }
}
