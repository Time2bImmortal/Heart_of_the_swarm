# Virtual Environment Simulator

The Virtual Environment Simulator is a powerful and flexible tool for creating closed-loop experiments involving multiple windows and Arduino integration. The simulator is designed to be easily customizable, allowing users to configure various parameters at the start, enabling the creation of original and tailored experiments.

## Overview

The simulator leverages the power of Python's multiprocessing to create independent processes for multiple windows, while also integrating with an Arduino board to perform closed-loop experiments. The configuration of the simulation parameters is done through the `parameters.py` file, which contains a dictionary (`P`) with nested fields specifying the desired settings.

The code structure is composed of several modules:

- `main.py`: Orchestrates the setup and execution of the virtual environment.
- `VirtualSimulation.py`: Defines the `Simulation` class, responsible for creating and managing the virtual environment.
- `ArduinoSerial.py`: Contains the `ArduinoDetector` class, which communicates with the Arduino board and handles closed-loop experiments.
- `parameters.py`: Stores the configuration dictionary `P`, which defines the simulation, structure, sensor, and arduino parameters.

## Customizable Parameters

Many parameters can be chosen at the start of the simulation to tailor the experiment to your specific needs. These parameters are defined in the `parameters.py` file and can be easily adjusted. Some of the key parameters include:

- Simulation direction (forward or backward)
- Number of sprites in the environment
- Velocity of movement
- Window positions
- Sensor threshold values
- Arduino code path
- Recording settings

By adjusting these parameters, you can create a wide range of experiments to meet your research or exploration objectives.

## Extending the Simulator

The Virtual Environment Simulator has been designed with extensibility in mind. Users can add custom code snippets or modify the existing code to create original experiments that meet their specific requirements.

For example, you might want to implement a new sensor type, add new visual elements to the environment, or create a more complex closed-loop control system. By studying the existing code and understanding how the various modules interact, you can easily extend the simulator's capabilities to suit your needs.

## Arduino integration

For more advanced use cases, such as changing the threshold values or modifying other parameters in the Arduino code, you will need to have the arduino-cli tool installed on your computer. The simulator can automatically make changes to the Arduino code based on the parameters specified in the parameters.py file. 

## Getting Started

To start using the Virtual Environment Simulator, follow these steps:

1. Ensure you have Python installed, along with the required libraries (e.g., `pygame`, `serial`, and `multiprocessing`).
2. Adjust the parameters in the `parameters.py` file to configure the simulation to your liking.
3. Run the `main.py` script to start the simulation.
4. If you have an Arduino board connected, first make sure that it is set up correctly. The Arduino_code_mouse folder contains the Arduino code required for the simulator. Before running the simulation, upload the code to the Arduino board using the Arduino IDE or a similar tool.

As you become more familiar with the simulator, you can explore its many features and customize the code to create unique and powerful experiments.
