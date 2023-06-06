# import pygame
# from pygame.locals import *
from ArduinoSerial import *
from VirtualSimulation import *
from multiprocessing import Process, Value
from parameters import P


"""This program sets up a virtual environment using the power of multiprocessing to create independent processes for 
multiple windows. 
Additionally, it allows integration with an Arduino board to perform closed-loop experiments. The parameters are 
specified in the 'parameters.json' file. """


def create_simulation(win_pos, simulation_params, flag, control_system):
    """Create a simulation with the specified window position, simulation parameters, flag, and control system."""
    simulation = Simulation(win_pos, control_system, **simulation_params, state=FULLSCREEN)
    simulation.run_experiment(flag)


def main(configuration):
    """The main function that orchestrates the setup and execution of the virtual environment."""

    # Obtain simulation and structure parameters from the configuration
    simulation_params = configuration["simulation_params"]
    structure_params = configuration["structure_params"]

    # Obtain simulation and structure parameters from the configuration
    win_positions = structure_params["window_position"]

    # Set up Arduino parameters, sensor threshold, file recording, and control systems
    arduino_params = configuration["sensor_params"]

    # Create a shared flag value for closed-loop control
    Flag = Value('i', False)

    # Determine the number of Arduino detector processes to create based on the sensor_params with the most values
    num_detectors = max(len(x) if isinstance(x, list) else 1 for x in arduino_params.values())

    # Create processes for the Arduino detectors
    detector_processes = []
    for i in range(num_detectors):
        # Update arduino_params for each process
        ard_params = {}
        for key, value in arduino_params.items():
            if isinstance(value, list):
                ard_params[key] = value[i] if i < len(value) else value[-1]
            else:
                ard_params[key] = value

        change_and_upload_threshold(ard_params["arduino_code_path"], ard_params["new_threshold_value"],
                                    change_threshold=ard_params["Change_threshold"])

        detector = ArduinoDetector(Flag, ard_params["recording_file"], port=ard_params["port"],
                                   baudrate=ard_params["baudrate"], time_flag=ard_params["time_flag_duration"],
                                   flag_duration=ard_params["flag_duration"])

        # Start recording if specified in the configuration
        if ard_params["Recording"]:
            detector.start(recording=True)

        # Start the detector process if a control system is chosen
        if ard_params["control_system"] != 0:
            p_detector = Process(target=detector.start)
            detector_processes.append(p_detector)
            p_detector.start()

    # Create processes for the simulation windows
    processes = []
    for i, win_pos in enumerate(win_positions):
        # Create a copy of simulation_params for each process
        sim_params = simulation_params.copy()

        # Update the simulation parameters with the corresponding values for each process
        for key, value in sim_params.items():
            if isinstance(value, list) and len(value) > i:
                sim_params[key] = value[i]

        # Create a new process for each window position
        p = Process(target=create_simulation, args=(tuple(win_pos), sim_params, Flag, arduino_params["control_system"]))
        processes.append(p)
        p.start()

    for p in processes + detector_processes:
        p.join()


if __name__ == "__main__":

    # Load the configuration from the JSON file and run the virtual environment
    pygame.init()
    main(P)
    pygame.quit()
    exit()
