import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from scipy.stats import shapiro, levene, friedmanchisquare, rankdata, f_oneway, ttest_rel
import pingouin as pg
import csv
import mplcursors
from math import atan2, degrees, pi
from tkinter import filedialog
import os
import pandas as pd


def choose_files():
    chosen_files = filedialog.askdirectory()
    files = []
    for filename in os.listdir(chosen_files):
        if filename.endswith('.csv'):
            files.append(filename)
    return chosen_files, files


def create_directories(chosen_files, parameters):

    directory = f'{chosen_files}/{parameters}'

    plot_xyz_folder, footprint_folder = f'{directory}/Plot_XYZ', f'{directory}/Footprint'

    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(plot_xyz_folder):
        os.makedirs(plot_xyz_folder)
    if not os.path.exists(footprint_folder):
        os.makedirs(footprint_folder)
    return directory, plot_xyz_folder, footprint_folder


def process_files(files, summary=False):
    data_dict = {}
    for file in files:
        # Process data_frame
        data_frame = data_arrangement(file)
        data_frame = extract_data(data_frame)
        if summary:
            data_frame = summarize_frames(data_frame)
        nested_dict = create_nested_dict(file, data_frame)
        data_dict = update_master_dict(data_dict, nested_dict)

    return data_dict


def sort_experiment_types(data_dict, order=['O', 'S', 'C']):
    """
    Sorts the experiment types for each subject in the given order.

    Args:
        data_dict (dict): The dictionary to sort.
        order (list): The desired order of the experiment types.

    Returns:
        dict: The sorted dictionary.
    """
    sorted_dict = {}
    for subject, experiments in data_dict.items():
        sorted_experiments = {}
        for experiment_type in order:
            if experiment_type in experiments:
                sorted_experiments[experiment_type] = experiments[experiment_type]
        sorted_dict[subject] = sorted_experiments
    return sorted_dict


def adjust_coordinates(coords):
    adjusted_coords = coords - coords[0]
    return adjusted_coords


def adjust_dataframe_coordinates(data_dict):
    for subject, subject_data in data_dict.items():
        for experiment_type, experiment_data in subject_data.items():
            df = experiment_data['dataframe']
            coords = df[['dis_y', 'dis_x']].to_numpy()
            x_coords = ball_radius * coords[:, 0]
            y_coords = ball_radius * coords[:, 1] * -1
            adjusted_coords = adjust_coordinates(np.column_stack((x_coords, y_coords)))

            df['dis_y'] = adjusted_coords[:, 0]
            df['dis_x'] = adjusted_coords[:, 1]
            data_dict[subject][experiment_type]['dataframe'] = df
    return data_dict


def data_arrangement(name):
    df = pd.read_csv(os.path.join(chosen_files, name))
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(df.index[:2])
    df.columns = [str(i + 1) for i in range(df.shape[1])]
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    return df


def extract_data(df):

    start = (df['25'] == 999).idxmax()
    df = df.loc[start:start+frames].copy()
    df.loc[(df['7'] >= 0.0) & (df['7'] <= lateral_positive), '7'] = 0.0
    df.loc[(df['5'] >= 0.0) & (df['5'] <= forward_backward), '5'] = 0.0
    df.loc[(df['7'] <= 0.0) & (df['7'] >= -lateral_negative), '7'] = 0.0
    df.loc[(df['5'] <= 0.0) & (df['5'] >= -forward_backward), '5'] = 0.0

    df.loc[:, ['5', '6', '7']] = df[['5', '6', '7']].rolling(window=10).median()

    df = df[['5', '6', '7', '14', '15']]
    df.columns = ['x', 'y', 'z', 'dis_x', 'dis_y']

    return df


def create_nested_dict(file_name, df):
    video_number, experiment_type, subject_number = file_name.split("_")
    subject_number = subject_number[:3]
    return {
        subject_number: {
            experiment_type: {
                'folder_name': chosen_files,
                'video_number': video_number,
                'dataframe': df
            }
        }
    }


def summarize_frames(df):
    df['dx'] = ball_radius * df['dx']
    df['dy'] = ball_radius* df['dy']
    grouped_df = [df.iloc[i:i + 25, :] for i in range(0, df.shape[0], 25)]
    summarized_df = pd.DataFrame(columns=['x', 'y', 'z', 'dx', 'dy'])
    for group in grouped_df:
        x = group['x'].sum()
        y = group['y'].sum()
        z = group['z'].sum()
        dx = group['dx'].sum()
        dy = group['dy'].sum()
        summarized_df = pd.concat([summarized_df, pd.DataFrame({'x': [x], 'y': [y], 'z': [z], 'dx': [dx], 'dy':[dy]})], ignore_index=True)
    return summarized_df


def dimensions_conversion(df):
    a, b, c = 0, 0, 1
    n = np.array([a, b, c])
    df_2d = pd.DataFrame(columns=['dx', 'dy'])
    for i in range(len(df)):
        # calculate projection
        x, y, z = df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['z']
        p = np.array([y, z, x]) - ((np.dot(np.array([y, z, x]), n) / np.dot(n, n)) * n)
        df_2d = pd.concat([df_2d, pd.DataFrame([[p[0], p[1]]], columns=['dx', 'dy'])], ignore_index=True)
    return df_2d


def animate_vectors(df):
    fig, ax = plt.subplots(figsize=(6,6))

    # initialize circle
    circle = plt.Circle((0,0), radius=1.2, fill=False)
    ax.add_artist(circle)
    ax.set_xlim((-1.2, 1.2))
    ax.set_ylim((-1.2, 1.2))
    colors = ['r'] + ['b'] * (len(df) - 1)
    # initialize vector
    # vector = ax.quiver([0], [0], [0], [0], angles='xy', scale_units='xy', scale=1)
    vector = ax.arrow(0, 0, df.iloc[0]['dx'], df.iloc[0]['dy'], color=colors[0], head_width=0.03, head_length=0.1)
    def update(frame):
        # get dx and dy for current row
        nonlocal vector
        vector.remove()
        vector = ax.arrow(0, 0, df.iloc[frame]['dx']*-1, df.iloc[frame]['dy']*-1, color=colors[frame], head_width=0.03,
                          head_length=0.1)
        return vector,

    ani = FuncAnimation(fig, update, frames=len(df), interval=1000, blit=True, repeat=True)
    plt.show()


def update_master_dict(data_dict, nested_dict):
    for subject_number, experiment_data in nested_dict.items():
        if subject_number not in data_dict:
            data_dict[subject_number] = {}
        for experiment_type, experiment_info in experiment_data.items():
            if experiment_type not in data_dict[subject_number]:
                data_dict[subject_number][experiment_type] = experiment_info
            else:
                data_dict[subject_number][experiment_type]['folder_name'] = experiment_info['folder_name']
                data_dict[subject_number][experiment_type]['video_number'] = experiment_info['video_number']
                data_dict[subject_number][experiment_type]['dataframe'] = experiment_info['dataframe']
    return data_dict


def calculate_straight_vs_turning(df, accuracy, threshold_angle):
    def calculate_angle(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        angle_rad = atan2(dy, dx)
        return degrees(angle_rad)

    straight_count = 0
    turning_count = 0
    total_segments = 0

    for i in range(0, len(df) - 2 * accuracy, accuracy):
        x1, y1 = df.iloc[i]['dis_y'], df.iloc[i]['dis_x']
        x2, y2 = df.iloc[i + accuracy]['dis_y'], df.iloc[i + accuracy]['dis_x']
        x3, y3 = df.iloc[i + 2 * accuracy]['dis_y'], df.iloc[i + 2 * accuracy]['dis_x']

        angle1 = calculate_angle(x1, y1, x2, y2)
        angle2 = calculate_angle(x2, y2, x3, y3)
        turning_angle = abs(angle2 - angle1)

        if turning_angle > 180:
            turning_angle = 360 - turning_angle

        if -threshold_angle <= turning_angle <= threshold_angle:
            straight_count += 1
        else:
            turning_count += 1

        total_segments += 1

    straight_percentage = (straight_count / total_segments) * 100
    turning_percentage = (turning_count / total_segments) * 100
    return straight_percentage, turning_percentage


def count_angles(df, col1='x', col2='y', angle_threshold=40):
    # Calculate angles in degrees
    angles = []
    # df[col2] = df[col2]
    # df[col1] = df[col1]
    df['x_component'] = ball_radius * np.sin(df[col1])  # y-axis radians to x-axis component
    df['y_component'] = ball_radius * np.sin(df[col2])  # z-axis radians to y-axis component
    for x, y in zip(df['x_component'], df['y_component']):
        if x == 0 and y == 0:
            angle = None
        else:
            angle = np.degrees(np.arctan2(y, x))
            if angle < 0:
                angle += 360
        angles.append(angle)

    df['angle'] = angles

    # Define angle ranges
    straight_lower = 90 - angle_threshold
    straight_upper = 90 + angle_threshold

    # Categorize angles based on threshold
    straight_mask = (df['angle'] >= straight_lower) & (df['angle'] <= straight_upper)
    right_mask = (df['angle'] > straight_upper) & (df['angle'] <= 360)
    left_mask = (df['angle'] < straight_lower) & (df['angle'] >= 0)
    rest_mask = pd.isna(df['angle'])

    # Create a separate column for angle labels
    df['angle_label'] = None
    df.loc[straight_mask, 'angle_label'] = 'straight'
    df.loc[right_mask, 'angle_label'] = 'lateral_r'
    df.loc[left_mask, 'angle_label'] = 'lateral_l'
    df.loc[rest_mask, 'angle_label'] = 'rest'
    return df


def categorize_motion(df, angle_col='angle_label', min_rest_chunk=20, max_interval=10):
    df['motion_rest'] = df[angle_col].apply(lambda x: 'motion' if x in ['straight', 'lateral_l', 'lateral_r'] else x)

    # Change the following line to use 'motion_rest' column
    rest_mask = (df['motion_rest'] == 'rest')

    rest_blocks = []
    rest_start = None
    for i in range(len(df)):
        if rest_mask.iloc[i]:
            if rest_start is None:
                rest_start = i
        else:
            if rest_start is not None:
                rest_blocks.append((rest_start, i))
                rest_start = None
    if rest_start is not None:
        rest_blocks.append((rest_start, len(df)))

    filtered_blocks = [block for block in rest_blocks if block[1] - block[0] >= min_rest_chunk]

    if not filtered_blocks:
        return df

    merged_blocks = [filtered_blocks[0]]
    for block in filtered_blocks[1:]:
        last_block = merged_blocks[-1]
        if block[0] - last_block[1] <= max_interval:
            merged_blocks[-1] = (last_block[0], block[1])
        else:
            merged_blocks.append(block)

    for block in merged_blocks:
        df.loc[block[0]:block[1] - 1, 'motion_rest'] = 'rest'

    return df


def count_motion_rest(df, motion_rest_col='motion_rest', min_rest_chunk=20):
    motion_count = (df[motion_rest_col] == 'motion').sum()
    rest_count = (df[motion_rest_col] == 'rest').sum()

    rest_blocks = 0
    in_rest_block = False
    rest_block_start = None

    for index, value in enumerate(df[motion_rest_col]):
        if value == 'rest' and not in_rest_block:
            in_rest_block = True
            rest_block_start = index
        elif value == 'motion' and in_rest_block:
            in_rest_block = False
            if index - rest_block_start >= min_rest_chunk:
                rest_blocks += 1

        # Check if the last rest block meets the minimum size requirement
    if in_rest_block and len(df) - rest_block_start >= min_rest_chunk:
        rest_blocks += 1

    return motion_count, rest_count, rest_blocks


def count_motion_angles(df):
    angle_counts = {'straight': 0, 'lateral_l': 0, 'lateral_r': 0, 'resting' : 0}

    for index, row in df.iterrows():
        if row['motion_rest'] == 'motion' and row['angle_label'] in angle_counts:
            angle_counts[row['angle_label']] += 1
    return angle_counts


def plot_motion_cgpt4(data_dict):
    folder = "Plots"
    if not os.path.exists(os.path.join(directory, folder)):
        os.makedirs(os.path.join(directory, folder))

    experiments = list(set(experiment_type for subject in data_dict for experiment_type in data_dict[subject].keys()))

    def plot_experiments(y_labels, title, filename, experiments, data_dict, sharex=True):
        fig, axes = plt.subplots(len(y_labels), 1, sharex=sharex)
        fig.suptitle(Trial_name+ str(frames) + '\n' + title)
        tick_labels = ['open', 'anti', 'with']
        # Ensure axes is always iterable
        if len(y_labels) == 1:
            axes = [axes]

        for y_label, ax in zip(y_labels, axes):
            y_vals = [data_dict[subject][experiment]['results'][y_label] for subject in data_dict for experiment in data_dict[subject]]
            for j, experiment in enumerate(experiments):
                y = [y_vals[k] for k in range(j, len(y_vals), len(experiments))]
                x = [j + np.random.normal(0, 0.02) for _ in range(len(y))]
                ax.set_ylabel(y_label)
                ax.boxplot(y, positions=[j], whiskerprops=dict(linestyle='-', linewidth=1.3), widths=1, medianprops={'color': 'black'})
                ax.scatter(x, y, s=3, c='black')

        axes[-1].set_xlabel('Experiment type')
        axes[-1].set_xticks(range(len(tick_labels)))
        axes[-1].set_xticklabels(tick_labels)

        plt.savefig(os.path.join(directory, folder, filename))

    plot_experiments(['motion_fraction'],
                     'Motion_fraction',
                     'Motion.jpg',
                     experiments,
                     data_dict)
    plot_experiments(['average_pause_duration [s]'], 'Average resting time [s]', "Resting.jpg", experiments, data_dict)

    plot_experiments(['lateral_motion', 'straight_motion'],
                     'Moving Direction',
                     'Direction.jpg',
                     experiments,
                     data_dict)

    plot_experiments(['pause_number'],
                     'Number of Pauses',
                     'Pauses_number.jpg',
                     experiments,
                     data_dict,
                     sharex=False)
    plot_experiments(['distance_walked'],
                     'Distance',
                     'Distance.jpg',
                     experiments,
                     data_dict,
                     sharex=False)


def create_csv(data_dict):
    filename = f"{directory}/output.csv"
    # Define the headers for the CSV file
    headers = ["subject number", "motion fraction", "resting motion", 'n_pauses', 'lateral_motion', 'distance_walked', "experiment type"]

    # Create a list to store the rows
    rows = []

    # Iterate through the data dictionary and add each row to the list
    for subject_number, experiments in data_dict.items():
        for experiment_type, experiment_data in experiments.items():
            # Extract the results dictionary
            values = experiment_data['results']

            # Extract the values for motion fraction, resting motion, and pause times
            motion_fraction = values.get("motion_fraction", None)
            resting_motion = values.get("average_pause_duration [s]", None)
            pause_times = values.get("pause_number", None)
            distance = values.get('distance_walked', None)
            side = values.get('lateral_motion', None)


            # Create a new row and add it to the list
            row = [subject_number, motion_fraction, resting_motion, pause_times, distance, side, experiment_type]
            rows.append(row)

    # Sort the rows by subject number and experiment type
    sorted_rows = sorted(rows, key=lambda x: (x[6], x[0]))

    # Write the sorted rows to the output CSV file
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(sorted_rows)


def compare_experiments(data_dict):

    output_file = os.path.join(directory, 'Results.txt')
    with open(output_file, 'w') as f:
        f.write(f"                                          _______________{Trial_name}_______________\n\n")
        f.write(f"Parameters:{parameters}\n")

        f.write("\nStatistical Analysis Summary:\n\n")
        f.write("1. Normality test (Shapiro-Wilk):\n")
        f.write("   We assessed the normality of the data for each of the three experiment groups.\n\n")
        f.write("2. Homogeneity of variance test (Levene):\n")
        f.write("   We tested if the three experiment groups had equal variances.\n\n")
        f.write("3. Friedman test:\n")
        f.write(
            "   We performed the Friedman test to compare the three experiment groups' outcomes across multiple dependent variables.\n\n")
        f.write("4. Post-hoc pairwise t-tests with Bonferroni correction:\n")
        f.write(
            "   To account for multiple comparisons and control the family-wise error rate, we conducted pairwise t-tests with Bonferroni   correction between the experiment groups.\n")

        # subjects = data_dict.keys()
        experiments = list(set([exp for subject_data in data_dict.values() for exp in subject_data.keys()]))
        outcomes = list(set([res for subject_data in data_dict.values() for exp_data in subject_data.values() for res in exp_data['results'].keys()  if res != "pos_2d_map"]))
        for outcome in outcomes:
            f.write(f"\n\nComparing experiments for {outcome}:\n\n")
            data_for_calc = []
            for experiment_type in experiments:
                exp_data = []
                for subject_data in data_dict.values():
                    if experiment_type in subject_data:
                        exp_data.append(subject_data[experiment_type]['results'][outcome])
                data_for_calc.append(exp_data)
            # Create NumPy arrays for the three experiment types
            exp_data_arr = [np.array(data_for_calc[i]) for i in range(len(experiments))]

            # Loop over each experiment type and perform statistical tests
            for i in range(len(experiments)):
                # Test for normality using Shapiro-Wilk test
                # f.write(f"Experiment {i + 1} ({experiments[i]}): {exp_data_arr[i]}\n")
                try:
                    norm_test = shapiro(exp_data_arr[i])
                    if norm_test[1] < 0.05:
                        f.write(f"Data for Experiment {i + 1} ({experiments[i]}) is not normally distributed    ")
                except Exception as e:
                    f.write(f"Error occurred during normality test for Experiment {i + 1} ({experiments[i]}): {e}")

                # Test for homogeneity of variance using Levene's test
                try:
                    levene_test = levene(*exp_data_arr)
                    if levene_test[1] < 0.05:
                        f.write(f"\nData for Experiment {i + 1} ({experiments[i]}) does not have equal variances     ")
                except Exception as e:
                    f.write(f"Error occurred during variance test for Experiment {i + 1} ({experiments[i]}): {e}\n")
            # Perform Friedman test on the data
            try:
                if len(data_for_calc) < 2:
                    raise ValueError("At least two experiments are required for the Friedman test")
                friedman_test = friedmanchisquare(*data_for_calc)
                if friedman_test[1] < 0.05:
                    # Calculate ranks of the data
                    ranks = np.apply_along_axis(rankdata, 0, data_for_calc)

                    # Calculate mean ranks for each experiment type
                    mean_ranks = np.mean(ranks, axis=1)

                    # Calculate differences between mean ranks
                    mean_rank_diffs = np.abs(np.subtract.outer(mean_ranks, mean_ranks))

                    # Perform pairwise t-tests with Bonferroni correction
                    posthoc_pvals = []
                    for i in range(len(experiments)):
                        for j in range(i + 1, len(experiments)):
                            t_stat, p_val = ttest_rel(data_for_calc[i], data_for_calc[j])
                            posthoc_pvals.append(p_val)

                    # Apply Bonferroni correction
                    posthoc_pvals = np.array(posthoc_pvals) * (len(posthoc_pvals) * (len(posthoc_pvals) - 1)) / 2

                    # Reshape p-values into matrix
                    posthoc_pval_matrix = np.zeros((len(experiments), len(experiments)))
                    posthoc_pval_matrix[np.triu_indices(len(experiments), k=1)] = posthoc_pvals
                    posthoc_pval_matrix = posthoc_pval_matrix + posthoc_pval_matrix.T

                    # Print results
                    # f.write(f"\n\nFriedman test: statistic={friedman_test[0]:.3f}, p-value={float(friedman_test[1]):.3f}     ")
                    # f.write(f"Mean ranks:{float(mean_ranks):.3f}    ")
                    # f.write(f"Mean rank differences:{float(mean_rank_diffs):.3f}")
                    f.write(
                        f"\n\nFriedman test: statistic={friedman_test[0]:.3f}, p-value={float(friedman_test[1]):.3f}\n")
                    f.write("Mean ranks: " + ', '.join([f"{val:.3f}" for val in mean_ranks]))
                    f.write("\nMean rank differences: ")

                    # Write mean_rank_diffs as a formatted string
                    for row in mean_rank_diffs:
                        f.write(' '.join([f"{val:.3f}  " for val in row]))

                    # f.write(f"\nPost-hoc pairwise t-tests with Bonferroni correction:\n{posthoc_pval_matrix}")
                    f.write("\n\nPost-hoc pairwise t-tests with Bonferroni correction:\n")

                    # Write the column labels (experiment groups)
                    f.write(" " * 10)  # Add padding to align with row labels
                    for i, exp in enumerate(experiments):
                        f.write(f"{exp:10}")
                    f.write("\n")

                    # Write the matrix values
                    for i, row in enumerate(posthoc_pval_matrix):
                        f.write(f"{experiments[i]:10}")  # Write the row label (experiment group)
                        for val in row:
                            f.write(f"{val:.4f}  ")  # Write the p-value with 6 decimal places and some padding
                        f.write("\n")

                    significance_level = 0.05
                    significant_pairs = []

                    for i in range(len(experiments)):
                        for j in range(i + 1, len(experiments)):
                            p_val = posthoc_pval_matrix[i, j]
                            if p_val < significance_level:
                                significant_pairs.append((experiments[i], experiments[j]))

                    if significant_pairs:
                        f.write('-------------------------------------')
                        f.write("\nSignificant comparisons:  ")
                        for pair in significant_pairs:
                            f.write(f"{pair[0]}:{pair[1]} is significant\n")
                            f.write('-------------------------------------')
                    else:
                        f.write("\nNo significant comparisons found.\n")

                else:
                    f.write(f"Friedman wasn't significant")
            except Exception as e:
                f.write(f"Error occurred during Friedman test: {e}")


def y_computation(data_dict, subjects=None):

    if subjects is None:
        subjects = data_dict.keys()

    experiments = list(set(experiment_type for subject in subjects for experiment_type in data_dict[subject].keys()))

    for subject_number, subject in data_dict.items():
        if subject_number not in subjects:
            continue

        for experiment_type, experiment_info in subject.items():
            if experiment_type not in experiments or 'dataframe' not in experiment_info:
                continue

            df = experiment_info['dataframe']

            total_motion = get_chunks(df, 'x', 'z')  # Compute the walking_fraction
            average_pause, pause_number = get_chunks_zeros(df, 'x', 'z')  # Compute the resting_fraction, and pause

            if total_motion != 0.0:
                # straight_motion, lateral_motion = calculate_straight_vs_turning(df, 10, 20)
                lateral_motion = chunks_side(df, 'x', 'z')
                straight_motion = 1.0 - lateral_motion
            else:
                straight_motion, lateral_motion = 0, 0
            if pause_number == 0:
                average_pause = 0

            distance_walked, pos_2d_map = sum_and_pos(df, 'dis_x', 'dis_y')

            experiment_info['results'] = {
                "motion_fraction": total_motion/frames,
                "lateral_motion": lateral_motion/frames,
                "straight_motion": straight_motion/frames,
                "average_pause_duration [s]": average_pause / 25.0,
                "pause_number": pause_number,
                "distance_walked": distance_walked,
                "pos_2d_map": pos_2d_map
            }

    return data_dict


def sum_and_pos(df, col1='dis_x', col2='dis_y'):
    numeric_df = df.select_dtypes(include='number')

    # Calculate the differences between consecutive rows
    diffs = numeric_df.diff().dropna()

    # Calculate the Euclidean distance for each consecutive pair of points
    euclidean_distances = np.sqrt(diffs[col1]**2 + diffs[col2]**2)

    # Calculate the total distance traveled
    total_distance_traveled = np.sum(euclidean_distances)

    final_position = (df[col1].iloc[-1], df[col2].iloc[-1])

    return total_distance_traveled, final_position


def get_chunks(df, col1, col2, min_chunk_size=10, inter=20):
    df = df.reset_index(drop=True)
    df['new_col'] = (df[col1] != 0.0) | (df[col2] != 0.0)
    chunks = []
    chunk_start = None
    i = 0
    while i < len(df):
        if df.at[i, 'new_col']:
            if chunk_start is None:
                chunk_start = i
        else:
            if chunk_start is not None:
                chunks.append((chunk_start, i))
                chunk_start = None
        i += 1

    if chunk_start is not None:
        chunks.append((chunk_start, len(df)))
    filtered_chunks = [chunk for chunk in chunks if chunk[1] - chunk[0] + 1 >= min_chunk_size]
    if not filtered_chunks:
        return 0
    else:
        total_size = sum(chunk[1] - chunk[0] + 1 for chunk in filtered_chunks)
        return total_size


def get_chunks_pos(df, col, min_chunk_size=10):
    df = df.reset_index(drop=True)
    df['positives'] = (df[col] > 0.0)
    chunks = []
    chunk_start = None
    i = 0
    while i < len(df):
        if df.at[i, 'positives']:
            if chunk_start is None:
                chunk_start = i
        else:
            if chunk_start is not None:
                chunks.append((chunk_start, i))
                chunk_start = None
        i += 1
    if chunk_start is not None:
        chunks.append((chunk_start, len(df)))
    filtered_chunks = [chunk for chunk in chunks if chunk[1] - chunk[0] + 1 >= min_chunk_size]
    total_size = sum(chunk[1] - chunk[0] + 1 for chunk in filtered_chunks)
    return total_size


def chunks_side(df, col1, col2, min_chunk_size=10):
    df = df.reset_index(drop=True)
    df['forward'] = df[col1].between(0.0, 0.004)
    df['side'] = (df[col2] != 0.0)
    chunks = []
    chunk_start = None
    i = 0
    while i < len(df):
        if df.at[i, 'side'] and not df.at[i, 'forward']:
            if chunk_start is None:
                chunk_start = i
        else:
            if chunk_start is not None:
                chunks.append((chunk_start, i))
                chunk_start = None
        i += 1
    if chunk_start is not None:
        chunks.append((chunk_start, len(df)))
    filtered_chunks = [chunk for chunk in chunks if chunk[1] - chunk[0] + 1 >= min_chunk_size]
    total_size = sum(chunk[1] - chunk[0] + 1 for chunk in filtered_chunks)
    return total_size


def get_chunks_zeros(df, col1, col2, min_chunk_size=20, inter=9):

    df = df.reset_index(drop=True)
    df['2_zeros_columns'] = (df[col1] == 0.0) & (df[col2] == 0.0)
    chunks = []
    chunk_start = None
    i = 0

    while i < len(df):
        if df.at[i, '2_zeros_columns']:
            if chunk_start is None:
                chunk_start = i
        else:
            if chunk_start is not None:
                chunks.append((chunk_start, i))
                chunk_start = None
        i += 1

    if chunk_start is not None:
        chunks.append((chunk_start, len(df)))

    filtered_chunks = [chunk for chunk in chunks if chunk[1] - chunk[0] + 1 >= min_chunk_size]
    combined_chunks = []
    prev_end = None
    inter_size = 0

    for chunk in filtered_chunks:
        if prev_end is not None and chunk[0] - prev_end - 1 < inter:
            inter_size += chunk[0] - prev_end - 1
            combined_chunks[-1] = (combined_chunks[-1][0], chunk[1])
        else:
            combined_chunks.append(chunk)
        prev_end = chunk[1]

    if not combined_chunks:
        return 0, 0
    else:
        total_size = sum(chunk[1] - chunk[0] + 1 for chunk in combined_chunks)
        return (total_size + inter_size) / len(combined_chunks), len(combined_chunks)


def plot_xyz(data_dict, plot_xyz_folder):
    ax_list = ['x', 'y', 'z']

    for subject_number, experiments in data_dict.items():
        for experiment_type, experiment_data in experiments.items():
            data = experiment_data['dataframe']
            title = f'{Trial_name}_{frames}\nSubject {subject_number} - {experiment_type}'

            fig, ax = plt.subplots(3, 1, sharex='all')
            fig.suptitle(f"3-axis motion analysis")
            for j in range(3):
                ax[j].plot(data[ax_list[j]])
                ax[j].set_ylabel(ax_list[j])
                ax[j].set_ylim(-0.03, 0.03)
                if j == 0 or j == 2:
                    ax[j].axhline(y=lateral_positive, color='crimson', linestyle='--')
                    ax[j].axhline(y=-lateral_negative, color='crimson', linestyle='--')
                else:
                    ax[j].axhline(y=forward_backward, color='crimson', linestyle='--')
                    ax[j].axhline(y=-forward_backward, color='crimson', linestyle='--')
                # ax[j].legend()
            ax[2].set_xlabel('Frames')
            fig.tight_layout()
            fig.canvas.manager.set_window_title(f'{title}')
            plt.savefig(os.path.join(f'{plot_xyz_folder}/{subject_number}_{experiment_type}.jpg'))
            plt.close(fig)


def create_map_position(data_dict, subjects=None):
    if subjects is None:
        subjects = data_dict.keys()

    experiments = set(experiment_type for subject in subjects for experiment_type in data_dict[subject].keys())

    points = []
    for subject_number, subject in data_dict.items():
        if subject_number not in subjects:
            continue

        for experiment_type, experiment_info in subject.items():
            if experiment_type not in experiments or 'results' not in experiment_info:
                continue

            pos_2d_map = experiment_info['results']['pos_2d_map']
            points.append((pos_2d_map, experiment_type))

    experiment_types = list(experiments)
    colors = [f'{c_col}' if color == 'C' else f'{s_col}' if color == 'S' else f'{o_col}' for color in experiment_types]

    fig, ax = plt.subplots()
    for pos, exp_type in points:
        color = colors[experiment_types.index(exp_type)]
        ax.scatter(pos[0], pos[1], c=color, alpha=0.15)

    # Calculate mean positions for each experiment type
    mean_positions = {}
    for exp_type in experiment_types:
        mean_positions[exp_type] = np.mean([pos for pos, et in points if et == exp_type], axis=0)

    # Add mean positions to the plot
    for exp_type, mean_pos in mean_positions.items():
        color = colors[experiment_types.index(exp_type)]
        ax.scatter(mean_pos[0], mean_pos[1], c=color, alpha=1, marker='X', s=100, linewidths=2, edgecolors='k')

    legend_handles = [
        mpatches.Patch(color=f'{c_col}', label='C'),
        mpatches.Patch(color=f'{s_col}', label='S'),
        mpatches.Patch(color=f'{o_col}', label='O'),
    ]

    ax.legend(handles=legend_handles, title='Experiment Type', loc='upper left')
    plt.axvline(0, c=(0, 0, 0, 0.2), ls='--')
    plt.axhline(0, c=(0, 0, 0, 0.2), ls='--')
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.title(f"Final position for {Trial_name}_{frames}")
    fig.canvas.manager.set_window_title(f'{Trial_name}')
    plt.savefig(os.path.join(f'{directory}/Final_position.jpg'))


def create_raw_file(data_dict, output_directory):
    with open(f'{output_directory}/Raw_y_computation.txt', 'w') as file:
        # Iterate through the subjects
        for subject_number, subject in data_dict.items():
            # Write the subject number
            file.write(f"{Trial_name}\n\n")
            file.write(f"O: open, S: anti, C: with\n")
            file.write(f"Subject {subject_number}:\n")
            # Iterate through the experiment types
            for experiment_type, experiment_info in subject.items():
                # Check if the experiment has 'results' key
                if 'results' not in experiment_info:
                    continue

                experiment_results = experiment_info['results']
                # Write the experiment type
                file.write(f"    {experiment_type}:\n")
                # Iterate through the experiment results
                for result_name, result_value in experiment_results.items():
                    formatted_value = f"({result_value[0]:.2f}, {result_value[1]:.2f})" if isinstance(result_value,
                                                                                                      tuple) else f"{result_value:.2f}"
                    file.write(f"        {result_name}: {formatted_value}\n")
                file.write("\n")


def plot_path(data_dict):

    for subject_number, subject_data in data_dict.items():
        for experiment_type, experiment_data in subject_data.items():
            adjusted_coords = experiment_data['dataframe'][['dis_x', 'dis_y']].to_numpy()

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.plot(adjusted_coords[:, 0], adjusted_coords[:, 1], color='white', linewidth=2)
            ax.scatter(adjusted_coords[:, 0], adjusted_coords[:, 1], color='white', marker='o', edgecolors='black')

            ax.set_xlim(min(adjusted_coords[:, 0]) - 1, max(adjusted_coords[:, 0]) + 1)
            ax.set_ylim(min(adjusted_coords[:, 1]) - 1, max(adjusted_coords[:, 1]) + 1)

            ax.set_facecolor('black')
            ax.grid(color='white', linestyle='--', linewidth=0.5)
            fig.canvas.manager.set_window_title(f'{subject_number}_{experiment_type}')
            plt.savefig(os.path.join(f'{footprint_folder}/{subject_number}_{experiment_type}.jpg'))
            plt.close(fig)


if __name__ == '__main__':

    global chosen_files, o_col, s_col, c_col, frames, ball_radius, Trial_name
    frames, ball_radius = 1000, 6
    o_col, s_col, c_col = "red", "blue", "green"
    chosen_files, files = choose_files()
    Trial_name = chosen_files.split('_')[2][5:]
    # Set parameters
    lateral_negative, lateral_positive, forward_backward = 0.004, 0.002, 0.002
    parameters = f'{frames}_{forward_backward}_{lateral_positive}_-{lateral_negative}'

    # Directories
    directory, plot_xyz_folder, footprint_folder = create_directories(chosen_files, parameters)

    # Nested dictionaries
    DATA = sort_experiment_types(process_files(files))
    DATA = adjust_dataframe_coordinates(DATA)
    # plot_path(DATA)

    #results computation and raw
    DATA = y_computation(DATA)
    create_raw_file(DATA, directory)
    create_csv(DATA)
    #
    # # results plot
    # plot_xyz(DATA, plot_xyz_folder)
    plot_motion_cgpt4(DATA)
    compare_experiments(DATA)
    create_map_position(DATA)
    #
    plt.close('all')

    # print(count_angles(data_dict_2['001']['O']['dataframe']))
    # animate_vectors(data_dict_2['001']['O']['dataframe'])
    # plt.show()

