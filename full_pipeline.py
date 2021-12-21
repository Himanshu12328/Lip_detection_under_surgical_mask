
import cv2 as cv
import os
import sys
import random
import math
import re
import time
import numpy as np
import matplotlib
from matplotlib import pyplot
import json
import os
import shutil
import zipfile
from tensorflow import keras 
from mrcnn.visualize import display_instances
from mrcnn import utils
import mrcnn.model
import tensorflow 
import h5py

from keras.preprocessing.image import img_to_array
# import keras.engine.topology as KE
import custom_config 

print(keras.__version__)
print(tensorflow.__version__)
print(h5py.__version__)

import custom_config 

# def draw_image_with_boxes(filename, boxes_list):
#      # load the image
#      #data = cv.imread(filename)
#      # plot the image
#      pyplot.imshow(data)
#      # get the context for drawing boxes
#      ax = pyplot.gca()
#      # plot each box
#      for box in boxes_list:
#           # get coordinates
#           y1, x1, y2, x2 = box
#           # calculate width and height of the box
#           width, height = x2 - x1, y2 - y1
#           # create the shape
#           rect = Rectangle((x1, y1), width, height, fill=False, color='red', lw=5)
#           # draw the box
#           ax.add_patch(rect)
#      # show the plot
#      pyplot.show()

model_path='/home/fearless/environments/cv_env3.7/'
inference_config = custom_config.InferenceConfig(1)
model = mrcnn.model.MaskRCNN(mode="inference",config=inference_config,model_dir=model_path)
model.load_weights(model_path + 'mask_rcnn_object_0005.h5', by_name=True)
class_names=['BG','mask']

corner_track_params= dict(maxCorners=300,qualityLevel=0.2,minDistance=2,blockSize=2)
lk_params=dict(winSize=(15,15),maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,10,0.03))

cap = cv.VideoCapture(0)
ret, frame = cap.read()
img = img_to_array(frame)
results = model.detect([img], verbose=0)
(results[0]['masks'])
y=(results[0]['masks']).astype("uint8")*255

x1=int(((results[0]['rois']).T)[1][0])
y1=int(((results[0]['rois']).T)[0][0])
        
x2=int(((results[0]['rois']).T)[3][0])
y2=int(((results[0]['rois']).T)[2][0])
        
rcnnOutput = cv.bitwise_and(frame, frame, mask=y)
cv.rectangle(frame,(x2,y2),(x1,y1),(0, 0, 0),2)
cropped=rcnnOutput[y1:y2,x1:x2]
resize_mask=cv.resize(cropped,(250,250))
cv.imshow("preivious frame",resize_mask)

prev_frame=resize_mask.copy()
prev_gray=cv.cvtColor(prev_frame ,cv.COLOR_BGR2GRAY)
prevPts=cv.goodFeaturesToTrack(prev_gray,mask=None,**corner_track_params)
mask = np.zeros_like(prev_frame)

while(1):
    ret, frame = cap.read()
    img = img_to_array(frame)
   
    results = model.detect([img], verbose=0)
    print("workings")
    
    (results[0]['masks'])
    
    y=(results[0]['masks']).astype("uint8")*255
    
    try:
        x1=int(((results[0]['rois']).T)[1][0])
        y1=int(((results[0]['rois']).T)[0][0])
        
        x2=int(((results[0]['rois']).T)[3][0])
        y2=int(((results[0]['rois']).T)[2][0])
        
        rcnnOutput = cv.bitwise_and(frame, frame, mask=y)
        cv.rectangle(frame,(x2,y2),(x1,y1),(0, 0, 0),2)
        cropped=rcnnOutput[y1:y2,x1:x2]
        resize_mask=cv.resize(cropped,(250,250))
    except:
        print("NO mask") 

    frame_copy=resize_mask.copy()
    prevPts=cv.goodFeaturesToTrack(prev_gray,mask=None,**corner_track_params)
    frame_gray=cv.cvtColor(frame_copy,cv.COLOR_BGR2GRAY)
    nextPts, status, err=cv.calcOpticalFlowPyrLK(prev_gray,frame_gray,prevPts,None,**lk_params)  
    good_new=nextPts[status==1]
    good_prev=prevPts[status==1] 

    for i, (new,prev) in enumerate(zip(good_new,good_prev)):
        x_new,y_new=new.ravel()
        #np method , flattenning out arry,  because we wanna use it to draww
        x_prev,y_prev=prev.ravel()
        x_new=int(x_new)
        y_new=int(y_new)
        x_prev=int(x_prev)
        y_prev=int(y_prev)
        mask=cv.line(mask,(x_new,y_new),(x_prev,y_prev),(0,255,0),3)
        frame_1=cv.circle(frame_copy,(x_new,y_new),8,(0,0,255),-1) #-1 to fill circle

    output = cv.add(frame_1, mask)
    cv.imshow("sparse optical flow", output)
    prev_gray=frame_gray.copy()
    prevPts=good_new.reshape(-1,1,2)

    cv.imshow("MAK",resize_mask)     
    #cv.imshow("full",frame)
    #cv.imshow("TEST",resize_mask)
    if(cv.waitKey(1) & 0xFF== ord('q')):
        break


cv.destroyAllWindows()
cap.release()