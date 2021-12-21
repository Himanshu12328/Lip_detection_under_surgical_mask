import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



face_cascade=cv.CascadeClassifier('/home/fearless/CV/udemy_opencv/CV_Python/DATA/haarcascades/haarcascade_frontalface_default.xml')

corner_track_params= dict(maxCorners=300,qualityLevel=0.2,minDistance=2,blockSize=2)
lk_params=dict(winSize=(15,15),maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,10,0.03))

def face_rectangle_adjusted(input_img):
    input_img_copy=input_img.copy()
    face_rectangle=face_cascade.detectMultiScale(input_img_copy,scaleFactor=1.2,minNeighbors=3)

    for (x,y,w,h) in face_rectangle:
        cv.rectangle(input_img_copy,(x,y),(x+w,y+h),color=(255,0,0),thickness=2)

    return input_img_copy

#code to give input frame to Lukas kanada 
cap=cv.VideoCapture(0)

ret,prev_frame=cap.read()
face_rectangle_for_LK=face_cascade.detectMultiScale(prev_frame,scaleFactor=1.2,minNeighbors=3)

list_rectangle=list(face_rectangle_for_LK)

x_list=list(map(lambda item: item[0], face_rectangle_for_LK))
y_list=list(map(lambda item: item[1], face_rectangle_for_LK))
w_list=list(map(lambda item: item[2], face_rectangle_for_LK))
h_list=list(map(lambda item: item[3], face_rectangle_for_LK))

print(type(x_list))

x_value=x_list[0]
y_value=y_list[0]
w_value=w_list[0]
h_value=h_list[0]

coordinates =face_rectangle_for_LK
print(coordinates)
print(x_value)
print(y_value)
print(w_value)
print(h_value)


mask_rectangle=np.zeros_like(prev_frame)
mask_rectangle[y_value:(y_value+h_value),x_value:(x_value+w_value)]=prev_frame[y_value:(y_value+h_value),x_value:(x_value+w_value)]

#start of LK

prev_gray=cv.cvtColor(mask_rectangle ,cv.COLOR_BGR2GRAY)


prevPts=cv.goodFeaturesToTrack(prev_gray,mask=None,**corner_track_params)
# ** this allows dictionary call in the function



while True:


    ret,frame =cap.read()

    frame_copy=frame.copy()

    face_rectangle_for_LK_1=face_cascade.detectMultiScale(frame_copy,scaleFactor=1.2,minNeighbors=3)

    print(face_rectangle_for_LK_1)
    x_list_1=list(map(lambda item: item[0], face_rectangle_for_LK_1))
    y_list_1=list(map(lambda item: item[1], face_rectangle_for_LK_1))
    w_list_1=list(map(lambda item: item[2], face_rectangle_for_LK_1))
    h_list_1=list(map(lambda item: item[3], face_rectangle_for_LK_1))

    print(type(x_list_1))

    x_value_1=x_list_1[0]
    y_value_1=y_list_1[0]
    w_value_1=w_list_1[0]
    h_value_1=h_list_1[0]

    coordinates =face_rectangle_for_LK_1
    print(coordinates)
    print(x_value_1)
    print(y_value_1)
    print(w_value_1)
    print(h_value_1)


    mask_rectangle_1=np.zeros_like(frame_copy)
    mask_rectangle_1[y_value_1:(y_value_1+h_value_1),x_value_1:(x_value_1+w_value_1)]=frame_copy[y_value_1:(y_value_1+h_value_1),x_value_1:(x_value_1+w_value_1)]
    cv.imshow('rectangle1111',mask_rectangle_1)

    mask=np.zeros_like(mask_rectangle_1)

    frame_gray=cv.cvtColor(mask_rectangle_1,cv.COLOR_BGR2GRAY)
    nextPts, status, err=cv.calcOpticalFlowPyrLK(prev_gray,frame_gray,prevPts,None,**lk_params) 
    # we are not giving nextPts here insted we are giving None, beacuse we want this fuction to return that
    #status outputs status vector where each element of the vector is set to 1 if flow coresponing features has been found, otherwise set to zero 

    good_new=nextPts[status==1]
    good_prev=prevPts[status==1]

    # ret,frame=cap.read()

    result3=face_rectangle_adjusted(mask_rectangle_1)

    cv.imshow('frame',result3)

    for i, (new,prev) in enumerate(zip(good_new,good_prev)):
        
        x_new,y_new=new.ravel()
        #np method , flattenning out arry,  because we wanna use it to draww
        x_prev,y_prev=prev.ravel()

        x_new=int(x_new)
        y_new=int(y_new)
        x_prev=int(x_prev)
        y_prev=int(y_prev)
        
        mask=cv.line(mask,(x_new,y_new),(x_prev,y_prev),(0,255,0),3)
        
        frame_1=cv.circle(mask,(x_new,y_new),8,(0,0,255),-1) #-1 to fill circle
        #to draw where the current point of drawing is

    img=cv.add(mask_rectangle_1,mask)
    frame[y_value_1:(y_value_1+h_value_1),x_value_1:(x_value_1+w_value_1)]=0
    img2=cv.add(frame,img)
    cv.imshow('tracking',img2)
    
    prev_gray=frame_gray.copy()

    prevPts=good_new.reshape(-1,1,2) # to get accepted in calcOpticalFlowPyrL

    if cv.waitKey(100) & 0xff==27:
        break



#mrcnn
#tensorflow 2.2.4
#keras 
#h5py2.10.0
cv.destroyAllWindows()
cap.release()