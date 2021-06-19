# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 21:30:19 2021

@author: Hussein Tofaili

Instructions:
Object tracking using cv2 library.
Once the first window pops, draw a rectangle around the object of interest
then hit the 'enter' button. The program should follow your object and draw
its velocity live as it goes.
While you are in the camera window, hit (and hold) the 'q' button to quit

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import misc

#start recording from the source:
#if you already have a video file, you can replace 0 with your file path
cap = cv2.VideoCapture(0) # 0 corresponds to the main webcam, 1 to a secondary webcam...

#choose a tracker:
tracker = cv2.TrackerBoosting_create()
#tracker = cv2.TrackerMOSSE_create()
#tracker = cv2.TrackerCSRT_create()

#read frames
success, img = cap.read()

bbox = cv2.selectROI("TRacking",img, False)
tracker.init(img, bbox)

#function to draw a box around the tracked object
def drawBox(img, bbox):
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.putText(img, "Tracking",(75,75), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2)

time=[]
trackx=[]
tracky=[]
count=0
px=0
py=0
count=0
fps0=0
velocity=[]

#live ploting of the object's velocity
#(use Qt5 for graphics backend if using Spyder notebook to get live update)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(np.linspace(0, 1*np.pi, 1000), np.linspace(0, 150*np.pi, 1000), 'r-')


while True:
    timer = cv2.getTickCount()
    success, img = cap.read()
    success, bbox = tracker.update(img)
    
    if success:
        
        drawBox(img, bbox)
        
        px = bbox[0] + bbox[2]/2
        py = bbox[1]+bbox[3]/2
        
        trackx.insert(count,px) # x position of tracked object
        tracky.insert(count,py) # y position of tracked object

        
    else:
        cv2.putText(img, "Lost",(75,75), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2)

    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(img, str(int(fps)),(75,50), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 255),2)
    cv2.imshow( "Tracking", img)
    
    if count==0:
        fps0=fps
    time.insert(count,count/fps0)
    
    if count>=1:
        #calculate the velocity (in the x-direction here)      
        f = InterpolatedUnivariateSpline(np.array(time),np.array(trackx),k=1)
        velocity=misc.derivative(f, np.array(time), dx=0.8, n=1) # in rad/s
        
        #Plot the time vs velocity
        line1.set_xdata(np.array(time))
        line1.set_ydata(np.array(velocity))
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.axis([0,np.array(time)[:-1].max()+10,np.array(velocity).min()-1,np.array(velocity).max()+1])
    
    count += 1
    
    #press (and hold) q to quit
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
    

