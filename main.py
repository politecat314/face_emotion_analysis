
from deepface import DeepFace
from helper_functions import makedir, progress_update, frame_to_int, emotions_dictionary_to_df
from matplotlib import pyplot as plt
from tqdm import tqdm

import cv2
import numpy as np
import os
import pandas as pd
import time

# SET THESE PARAMETERS YOURSELF
parent_dir = 'Data/Donald Trump' # dir where your video is located
video_name = 'Donald Trump VICTORY SPEECH _ Full Speech as President Elect of the United States.mp4' # name of the video
reference_image = f"{parent_dir}/trump.jpg" # if face is the same one we are looking for, then do emotion analysis


video_path = f"{parent_dir}/{video_name}"
output_dir = f"{parent_dir}/frames/{video_name}" # save frames here
output_csvs = f"{parent_dir}/output" # save output csv data here

makedir(f"{parent_dir}/frames")
makedir(output_dir)
makedir(output_csvs)
print()


cap = cv2.VideoCapture(video_path)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # number of frames in the video
print("total num of frames:", num_frames)
print("beginning frame extraction")

vidcap = cv2.VideoCapture(video_path)
success,image = vidcap.read()
count = 0

start_time = time.time() # measure estimated time to complete
total_frames = 0


while success:
    if count % 10 == 0: # skip every x frames
        cv2.imwrite(f"{output_dir}/frame_{count}.jpg", image)     # save frame as JPEG file
        total_frames += 1
        
    success, image = vidcap.read()
    
    count += 1
    if count % 100 == 0:
        progress_update(start_time, count, num_frames)

progress_update(start_time, count, num_frames)

print()
print(total_frames, "frames extracted")
print("frame extraction complete")
print()


# extracting frames which match face
print(f"extracting frames which match the face in {reference_image}")
verified_frames = []

for frame in tqdm(os.listdir(output_dir)):
    frame_path = f"{output_dir}/{frame}"

    # check if trump face is in frame
    result = DeepFace.verify(img1_path = frame_path, 
                         img2_path = reference_image, 
                         enforce_detection = False, # give false when failed to detect face
    )

    if result['verified']:
        verified_frames.append(frame)

print("Number of frames with matching face", len(verified_frames))

df = pd.DataFrame(verified_frames, columns=['verified_frame'])
df.to_csv(f"{parent_dir}/output/{video_name}_face_verified.csv", index=False)
print(f"finished extracting faces which match with face in {reference_image}")
print()


print("begin emotion analysis")
# store the emotions in a dictionary
emotions_dict = {}

start = time.time()
count = 0

num_frames = len(verified_frames)

for frm in verified_frames:
    # analyze the emotion for this frame
    frm_path = f"{output_dir}/{frm}"
    
    obj = DeepFace.analyze(img_path = frm_path, actions = ['emotion'], enforce_detection=False)
    emotions_dict[frm] = obj
    
    count += 1
    progress_update(start_time, count, num_frames)

progress_update(start_time, count, num_frames)
print("finish emotion analysis")

# convert emotions dictionary into a dataframe
df = emotions_dictionary_to_df(emotions_dict)
df.to_csv(f"{parent_dir}/output/{video_name}_emotions.csv", index=False)


# convert frame names to integers for sorting
df['int_name'] = df['frame_name'].map(frame_to_int)
df = df.sort_values(by=['int_name'])

# DRAW AND SAVE THE GRAPH
# find mean emotion of every x frames
aggregation = 50

mean_df = df.groupby(np.arange(len(df))//aggregation).mean()

mean_df.iloc[::].plot(x ='int_name', y=['angry',
 'disgust',
 'fear',
 'happy',
 'sad',
 'surprise',
 'neutral'], kind = 'line')

plt.title(f"average emotion every {aggregation} frames", fontsize=18)
plt.xlabel('frame #', fontsize=15)
plt.ylabel('probability', fontsize=15)
plt.rcParams["figure.figsize"] = (16,11)
plt.savefig(f"{output_csvs}/{video_name}_emotion_graph.jpg")
plt.show()

print(f"Finished. Graph saved in {output_csvs}/{video_name}_emotion_graph.jpg")