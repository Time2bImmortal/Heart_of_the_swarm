import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import numpy as np


def read_data(files, variable):
    data = {}
    for file in files:
        with open(file, 'r') as f:
            content = f.readlines()

        subject_data = []
        process_data = False  # Add this line to control when to process data
        for line in content:
            line = line.strip()
            if line.startswith("Subject"):
                subject = line
            elif line.startswith("O:"):
                process_data = True  # Process data only when the line starts with "O:"
            elif line.startswith("S:") or line.startswith("R:") or line.startswith("P:"):
                process_data = False  # Stop processing data for other letters
            elif line.startswith(variable) and process_data:
                key, value = line.split(':')
                subject_data.append(float(value.strip()))
        data[subject] = subject_data

    return data



def create_boxplots(data, variable, n_pairs):

    fig, ax = plt.subplots()
    data_to_plot = []
    labels = []
    scatter_positions = []

    for subject, subject_data in data.items():
        data_to_plot.append(subject_data)
        labels.append(subject)

    positions = []
    for i in range(n_pairs):
        positions.extend([2*i + 1, 2*i + 2])

    # Make the boxplot wider
    box_width = 0.6
    boxplot = ax.boxplot(data_to_plot, positions=positions, widths=box_width)

    # Add scatter points
    for i, (pos, subject_data) in enumerate(zip(positions, data_to_plot)):
        y = subject_data
        x = np.random.normal(pos, box_width / 6, size=len(y))
        ax.scatter(x, y, alpha=0.5, label=labels[i])

    ax.set_ylabel(variable)
    ax.set_xticks(positions)

    # Create two-level labels
    couple_names = ["Yossef", "Itai"]
    main_labels = []
    for name in couple_names:
        main_labels.extend([f"{name}\nForward", f"{name}\nBackward"])

    ax.set_xticklabels(main_labels)

    plt.title(f'Comparison of {variable}')
    plt.show()


def select_files():
    root = tk.Tk()
    root.withdraw()
    filepaths = filedialog.askopenfilenames()
    return list(filepaths)


def select_couples(n_couples):
    all_files = []
    for _ in range(n_couples):
        couple_files = []
        for i in range(2):
            print(f"Select file {i + 1} for couple {_ + 1}")
            file = filedialog.askopenfilename()
            if not file:
                raise ValueError("You must select a file.")
            couple_files.append(file)
        all_files.extend(couple_files)
    return all_files


# Choose the number of couples
n_couples = 2  # Replace with the desired number of couples

# Choose the input files
files = select_couples(n_couples)

# Choose the variable
variable_list = ['motion_fraction', 'lateral_motion', 'straight_motion', 'average_pause_duration [s]', 'pause_number', 'distance_walked']
variable = variable_list[5]

# Read data from files
data = read_data(files, variable)

# Create boxplots
create_boxplots(data, variable, n_couples)
