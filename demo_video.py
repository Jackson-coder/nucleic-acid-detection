import copy
import numpy as np
import cv2
from glob import glob
import os
import argparse
import json
import torch

# video file processing setup
# from: https://stackoverflow.com/a/61927951
import argparse
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple
import warnings
import time
from torch_utils import time_synchronized 
# openpose setup
from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

def process_frame(frame, body=True, hands=True):
    flag = False
    canvas = copy.deepcopy(frame)
    if body:
        T = time.time()
        candidate, subset = body_estimation(frame)
        nose = [] 
        r = []
        n = []
        le = []
        re = []
        T0 = time.time()
        for j in range(len(subset)):
            if subset[j][0] != -1:
                n = candidate[int(subset[j][0])][:3] #x,y,score
            if subset[j][16] != -1:
                re = candidate[int(subset[j][16])][:3]
            if subset[j][17] != -1:
                le = candidate[int(subset[j][17])][:3]

            if len(n)==0 or len(re)==0 or len(le)==0:
                warnings.warn('Please point the camera!!!')
                hands = False
            else:
                nose.append(n)
                r.append(le[0] - re[0])
        T1 = time.time()
        if r != []:
            index = r.index(max(r))
            center = nose[index]
            radius = max(r)
            subset = subset[index].reshape(1,-1)
        else:
            warnings.warn('Please point the camera!!!')
            hands = False
        T2 = time.time()
        # print("T0-T:",T0-T)
        # print("T1-T0:",T1-T0)
        # print("T2-T1:",T2-T1)
        # canvas = util.draw_bodypose(canvas, candidate, subset)
    if hands:
        # t_3 = time_synchronized()
        hands_list = util.handDetect(candidate, subset, frame)
        T3 = time.time()
        # t_4 = time_synchronized()
        # print(f'frame {frame_id} is Done. ({t_4 - t_3:.3f}s)')
        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(frame[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            if w/2 < radius*2 and peaks[0][1]>center[1]:
                delta = abs(peaks - center[:2])
                delta = np.sum(delta, axis=1)
                if sum(delta[delta[:]<radius])>5:
                    flag = True

            all_hand_peaks.append(peaks)
        T4 = time.time()    
        # print("T3-T2:",T3-T2)
        # print("T4-T3:",T4-T3)
        
        # canvas = util.draw_handpose(canvas, all_hand_peaks)
    return canvas, flag

# writing video with ffmpeg because cv2 writer failed
# https://stackoverflow.com/questions/61036822/opencv-videowriter-produces-cant-find-starting-number-error
import ffmpeg

# open specified video
parser = argparse.ArgumentParser(
        description="Process a video annotating poses detected.")
parser.add_argument('file', type=str, help='Video file location to process.')
parser.add_argument('--no_hands', action='store_true', help='No hand pose')
parser.add_argument('--no_body', action='store_true', help='No body pose')
parser.add_argument('--no_show', action='store_true', help='No body pose')

args = parser.parse_args()
video_file = args.file
show = not args.no_show
cap = cv2.VideoCapture(video_file)

# writer = None
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if show:
    writer = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w//2, h//2))
frame_id = 0

t0 = time.time()
time_buffer=[]
flag = False
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    frame_id+=1
    frame = cv2.resize(frame, (w//2, h//2))
    
    if frame_id % 20 != 0:
        if show:
            if flag:
                cv2.putText(frame, 'detecting', (10, 200), cv2.FONT_HERSHEY_TRIPLEX, 4, (255, 255, 0), 8)
            writer.write(frame)
        continue
    
    # print(frame.shape, w,h)
    
    # t1 = time_synchronized()
    posed_frame, flag = process_frame(frame, body=not args.no_body,
                                       hands=not args.no_hands)
    # t2 = time_synchronized()
    # print(f'frame {frame_id} is Done. ({t2 - t1:.3f}s)')
    # time_buffer.append(t2-t1)

    if show:
        # cv2.imshow('frame', posed_frame)
        if flag:
             cv2.putText(posed_frame, 'detecting', (10, 200), cv2.FONT_HERSHEY_TRIPLEX, 4, (255, 255, 0), 8)
        writer.write(posed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f'Done. ({time.time() - t0:.3f}s)')
# print(f'Model average inference: ({np.mean(time_buffer)}s)')
print('Average inference:', f'({(time.time() - t0)/frame_id:.3f}s)')

cap.release()
if show:
    writer.release()
cv2.destroyAllWindows()
