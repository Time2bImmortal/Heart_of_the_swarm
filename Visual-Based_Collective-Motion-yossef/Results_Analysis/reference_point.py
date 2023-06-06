from tkinter.filedialog import askdirectory
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from PIL import ImageChops
from tqdm import tqdm
# Ask the user to choose a directory containing video and data files
directory = askdirectory()
if not directory:
    print('No directory chosen.')
    quit()

# Find all .csv files in the chosen directory
data_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

# Process each data file and its associated video file
for data_file in data_files:
    print('Processing file:', data_file)

    # Open the associated video file
    video_file = os.path.splitext(data_file)[0] + '.avi'
    if not os.path.exists(video_file):
        print('No video file found for', data_file)
        continue

    # Suppress the first two rows of the data file, and rename columns to numbers
    data = pd.read_csv(data_file, header=2, names=range(25))

    # Process video file to find frames with high contrast
    source = cv2.VideoCapture(video_file)
    Frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))

    contrast_list = []

    for i in tqdm(range(2000)):
        ret, img = source.read()
        if not ret:
            print("Failed to read frame")
            continue
        if img.size == 0:
            print("Empty image")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_a = Image.fromarray(gray)
        if i > 0:
            diff = ImageChops.difference(gray_a, gray_b)
            diff = np.mean(np.asarray(diff))
            contrast_list.append(np.mean(diff))

        gray_b = gray_a

    # Find contrast frames and ask user to choose one
    contrast_threshold = 10
    contrast_frames = [(int(contrast_list[i]), i, i / 25) for i in range(len(contrast_list)) if contrast_list[i] > contrast_threshold]
    print('Found', len(contrast_frames), 'frames with contrast greater than', contrast_threshold)
    for i in contrast_frames:
        print(i)

    chosen_frame = None
    while chosen_frame is None:
        choice = input('Choose a frame by entering its index, or enter "skip" to skip this file: ')
        if choice == 'skip':
            break
        try:
            index = int(choice)
            if index >= 0 and index < len(contrast_frames):
                chosen_frame = contrast_frames[index]
            else:
                print('Invalid choice.')
        except ValueError:
            print('Invalid choice.')

    if chosen_frame is not None:
        # Add a column to the data file and mark the chosen row
        data[25] = np.nan
        data.at[chosen_frame[1]+500, 25] = 999
        print('Marked row', chosen_frame[1]+500, 'as 999')

        # Save the modified data file
        output_file = os.path.splitext(data_file)[0] + '.csv'
        data.to_csv(output_file, index=False)
        print('Saved modified data file to', output_file)

    cv2.destroyAllWindows()
    source.release()

    # Ask user if they want to process the next file
    choice = input('Do you want to process the next file? (y/n) ')
    if choice.lower() != 'y':
        break
