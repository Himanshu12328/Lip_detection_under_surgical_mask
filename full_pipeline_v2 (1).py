import cv2 as cv
import numpy as np
from matplotlib import pyplot
from tensorflow import keras 
from mrcnn.visualize import display_instances
from mrcnn import utils
import mrcnn.model
import tensorflow 
import h5py
from keras.preprocessing.image import img_to_array


import custom_config 

#Custom model path 
model_path=' ' 
inference_config = custom_config.InferenceConfig(1)

#Loading the model
model = mrcnn.model.MaskRCNN(mode="inference",config=inference_config,model_dir=model_path)
model.load_weights(model_path + 'mask_rcnn_object_0005.h5', by_name=True)

lk_params=dict(winSize=(15,15),maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,10,0.03))
#Here we are creating directory to pass in calcOpticalFlowPyrLK function later in this code

#Initilising the webcam stream
cap = cv.VideoCapture(0)
ret, frame = cap.read()


#Denoising the image
dst=cv.fastNlMeansDenoisingColored(frame,None,10,10,7,21)

#Converting image to array
img = img_to_array(dst)

#Face mask detection
results = model.detect([img], verbose=0)

#Creating mask for image segementation 
y=(results[0]['masks']).astype("uint8")*255

#Coordinates of rectangle around the face mask
x1=int(((results[0]['rois']).T)[1][0])
y1=int(((results[0]['rois']).T)[0][0])     
x2=int(((results[0]['rois']).T)[3][0])
y2=int(((results[0]['rois']).T)[2][0])

# image segementation        
rcnnOutput = cv.bitwise_and(frame, frame, mask=y)
#Drawing rectangle around the face mask
cv.rectangle(frame,(x2,y2),(x1,y1),(0, 0, 0),2)

#Cropping and resizing the ROI 
cropped=rcnnOutput[y1:y2,x1:x2]
resize_mask=cv.resize(cropped,(250 ,250))

cv.imshow("Initial_frame",resize_mask)

#Creating copy and coverting to grey scale
prev_frame=resize_mask.copy()
prev_gray=cv.cvtColor(prev_frame ,cv.COLOR_BGR2GRAY)

# ORB for feature  detections 
orb = cv.ORB_create()
kp = orb.detect(prev_gray,None)
prev_array = np.asarray([[[p.pt[0], p.pt[1]]] for p in kp])
prevPts = np.float32(prev_array)

# mask to draw the features on 
mask = np.zeros_like(prev_frame)


while(1):
    ret, frame = cap.read()
    #Denoising the image
    dst=cv.fastNlMeansDenoisingColored(frame,None,10,10,7,21)

    #Converting image to array
    img = img_to_array(dst)

    #Face mask detection
    results = model.detect([img], verbose=0)

    #Creating mask for image segementation     
    y=(results[0]['masks']).astype("uint8")*255
    
    try:
        #Coordinates of rectangle around the face mask
        x1=int(((results[0]['rois']).T)[1][0])
        y1=int(((results[0]['rois']).T)[0][0])
        x2=int(((results[0]['rois']).T)[3][0])
        y2=int(((results[0]['rois']).T)[2][0])
        # image segementation 
        rcnnOutput = cv.bitwise_and(frame, frame, mask=y)
        
        #Drawing rectangle around ROI
        cv.rectangle(frame,(x2,y2),(x1,y1),(0, 0, 0),2)
        
        #Cropping and resizing the ROI 
        cropped=rcnnOutput[y1:y2,x1:x2]
        resize_mask=cv.resize(cropped,(250 ,250))
    except:
        #Incase of no mask detected 
        print("NO MASK") 

    frame_copy=resize_mask.copy()

    #orb the detect the features 
    #kp is for the feature points
    kp = orb.detect(prev_gray,None)
    prev_array = np.asarray([[[p.pt[0], p.pt[1]]] for p in kp])
    prevPts = np.float32(prev_array)



    # to make image grayscale
    frame_gray=cv.cvtColor(frame_copy,cv.COLOR_BGR2GRAY)
    nextPts, status, err=cv.calcOpticalFlowPyrLK(prev_gray,frame_gray,prevPts,None,**lk_params)

    #** will allow us to pass directory that we created at the start of this code  
    # prev_gray first 8-bit input image
    # frame_gray second input image 
    # prevPts vector of 2D points for which the flow needs to be found
    # nextPts output vector of 2D points containing the calculated new positions of input features in the second image
    #we are not giving next points here insted we are giving None, because we want this fuction to return that.
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
    #To add both images to display mask and features drawn.

    cv.imshow("sparse optical flow", output)
    prev_gray=frame_gray.copy()

    prevPts=good_new.reshape(-1,1,2)  #to condition the output image to pass as a previous frame

    cv.imshow("MAK",resize_mask)     
    #cv.imshow("full",frame)
    #cv.imshow("TEST",resize_mask)
    if(cv.waitKey(1) & 0xFF== ord('q')):
        break


cv.destroyAllWindows()
cap.release()