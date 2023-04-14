This section discusses the process of extracting data from video recordings of our experiments. At present, the procedure is based on FicTrac, but you are encouraged
to enhance it significantly. 

The preprocessing.py script converts .avi video files to mp4 format (note the video size). It then prompts for the video number and opens each video file with an 
input window where you can enter the experiment type (modifiable in the code). This process renames each video accordingly. Following this step, FicTrac analyzes 
each video.

The reference_point.py script identifies areas with high luminosity differences. Based on your pre-experiment settings, you can define the start point (which creates
a column and writes 999). 
Finally, the results.py script creates a folder with a specific threshold and a certain number of frames. Inside this folder, various graphs and data analyses are
generated.

The final results is just about comparing two different experiments run, it was used to compare my results to Bleichman et al. 
