import os
import time
import tkinter as tk
from tkinter import filedialog
import threading
import pandas as pd
import subprocess
import cv2
import queue
from tqdm import tqdm
import csv

def convert_dat_to_csv(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".dat"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, delimiter=',')
            df.to_csv(os.path.join(folder_path, filename[:9]+'.csv'), index=False)
            print(f"Converted {filename} to {filename[:9]}.csv")

choices=["Open-loop", "Closed-loop straight", "Closed-loop crossed"]
def rename_file(file_path, experiment, subject):
    file_path = str(file_path)
    print(file_path)
    global start_num
    # filename = os.path.basename(file_path)
    new_name = f"{str(start_num).zfill(3)}_{experiment.upper()}_{str(subject).zfill(3)}.avi"
    start_num += 1
    new_name = new_name.replace("OPEN-LOOP", "O")
    new_name = new_name.replace("CLOSED-LOOP STRAIGHT", "S")
    new_name = new_name.replace("CLOSED-LOOP CROSSED", "C")
    directory = os.path.dirname(file_path)
    new_path = os.path.join(directory, new_name)
    new_path = new_path.replace(r"\0", "/0")
    print("new_path", new_path)
    os.replace(file_path, new_path)
    path_of_modified_files.append(new_path)

    return start_num


def tkinter_window(file_path, q, choices):
    root = tk.Tk()
    root.geometry("200x150+650+150")
    root.title("Define the video to my right")
    experiment = tk.StringVar(value=choices[0])
    subject = tk.StringVar()
    tk.Label(root, text="Experiment").grid(row=0)
    tk.Label(root, text="Subject").grid(row=1)
    tk.OptionMenu(root, experiment, *choices).grid(row=0, column=1)
    tk.Entry(root, textvariable=subject).grid(row=1, column=1)
    tk.Button(root, text="Submit", command=lambda:
    [q.put("stop"), time.sleep(2.5), rename_file(file_path, experiment.get(), subject.get())
        , root.destroy()]).grid(row=2, column=1)
    root.mainloop()


def open_video(file_path, q):
    file_path = file_path.replace("/", '\\',3)
    flag = True
    print(file_path)
    if os.path.exists(file_path):
        cap = cv2.VideoCapture(file_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1500)
        width = int(cap.get(3))
        height = int(cap.get(4))
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video", 400, 400)
        cv2.moveWindow("Video", 860, 0)
        while (cap.isOpened()) and flag:
            ret, frame = cap.read()
            if not q.empty():
                command = q.get()
                if command == "stop":
                    cap.release()
                    cv2.destroyAllWindows()
                    flag = False
            if ret:
                cv2.imshow("Video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q') or not q.empty():
                    flag = True
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"{file_path} not found.")


print("Choose a folder, and let's start working :)")
folder_path = filedialog.askdirectory()
Jump = input("Files to convert ? y/n")
if Jump == 'y':

    try:
        for filename in os.listdir(folder_path):
            if not filename.endswith(".avi"):
                filepath = os.path.join(folder_path, filename)
                new_filename = (filename.rsplit(".", 1)[0] + "-converted.avi").replace("C", "")
                new_filepath = os.path.join(folder_path, new_filename)
                subprocess.run(['ffmpeg', '-i', filepath, '-s', '1280x720', new_filepath])
                print("files converted")
    except FileNotFoundError:
        print("No corresponding file found.")

answer = input("Do you want to delete files without -converted in name? (y/n)")

if answer == "y":
    for filename in os.listdir(folder_path):
        if '-converted' not in filename:
            file_path = os.path.join(folder_path, filename)
            confirm = input(f"Do you want to delete {filename}? (y/n)")
            if confirm == "y":
                os.remove(file_path)
                print(f"{filename} deleted!")
            else:
                print(f"{filename} not deleted!")
                continue
else:
    print("No files were deleted!\n")
print("""This is the end of the first step!!! Now, we must define each video """)
namingfile = input("Do you want to define the recordings? y/n")
if namingfile =="y":
    global start_num
    start_num = int(input("Enter the starting number: "))
    global path_of_modified_files
    path_of_modified_files =[]

    avi_files = filter(lambda x: x.endswith('.avi'), os.listdir(folder_path))
    num_avi_files = len(list(avi_files))
    counter = 0
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith("ed.avi") and counter < num_avi_files:
            counter += 1
            file_path = f'{folder_path}' + f'/{filename[:]}'
            # file_path = file_path.replace("01/0", "01/00")
            q = queue.Queue()
            video_thread = threading.Thread(target=open_video, args=(file_path, q), daemon=True)
            tkinter_thread = threading.Thread(target=tkinter_window, args=(file_path, q, choices))
            video_thread.start()
            tkinter_thread.start()
            video_thread.join()
            tkinter_thread.join()
            next = input("Continue? y/n")
            if str(next) == "y":
                continue
            else:
                break
else:
    pass

print("We can now use fictrac to get a data file from the converted video")
response = input("Do you want to continue? y/n")
new_path = []
for name in os.listdir(folder_path):
    if len(name) == 13:
        path = os.path.join(folder_path, name)
        new_path.append(path)
print(new_path)
if response == 'y':
    # Moving directory
    fictrac_home = "C://Users/scr/vcpkg/fictrac/bin/Release"
    os.chdir("..")
    os.chdir(fictrac_home)
    print(f"you are in'{fictrac_home}'")
    for path in new_path:
        with open("config.txt", "r") as file:
            lines = file.readlines()
            src_fn_line = [line for line in lines if 'src_fn' in line][0]
            index = lines.index(src_fn_line)
            src_fn_line = src_fn_line.split(":")[0] + f': {path}\n'
            lines[index] = src_fn_line
        with open('config.txt', 'w') as file:
            file.writelines(lines)

        # subprocess.run(["cmd", 'start', '/wait'])
        # subprocess.run(["runas", "/user:Yossef Aidan", "cmd", "start", "/B","/wait", "configGui config"])
        # subprocess.run(["cmd", '/c', "configGui config.txt"])
        # subprocess.Popen(["cMd", "/wait", "configgui config"])
        subprocess.run(["cmd", '/c', 'start', '/B', '/wait', 'fictrac'])
else:
    print("Exiting the script.")


user_input = input("Do you want to convert .dat files to .csv files? (yes/no): ").lower()

if user_input == "y":

    print("Please choose the folder containing the .dat files.")
    # folder_path = filedialog.askdirectory()
    convert_dat_to_csv(folder_path)

"""It will great to create a new folder called clean data files to order the .csv files"""
