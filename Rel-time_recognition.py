# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 20:30:28 2018

@author: trandinhson3086
"""

import zmq
from array import array
import io
import cv2
import numpy as np
import os
from keras import models
import time
import requests

def main():

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    
    print("Connecting to: ", "localhost:5555")
    socket.bind("tcp://*:5555")
    
    with  open('.\models\model2021.json') as model_file:
        model = models.model_from_json(model_file.read())
        
    model.load_weights('.\models\model2021.hd5')
    final_labels = ['meaningless', 'Swiping Right' ,'Swiping Left' ,'Swiping Up','Swiping Down' ,'Zoom In', 'Zoom Out']
    print("Load done")
    
    count = 0
    
    frames = []
    X = []
    
    nframe = 30
    skip_frame = 5
    depth = 20

    pred_label = "Hello"
    while True:
        message = socket.recv().decode("utf-8")
        #print(message)
        if message == "STOP":
            print("Receive stop")
            exit()
        data = cv2.imread(message)
        gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY) 
        data = cv2.resize(data, (32,32))
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY) 
        frames.append(data)
        
        
        # if len(frames) > nframe:
        #     for i in range(5):
        #         frames.pop(0)
        
        

        if len(frames) == nframe:
            frames_arr = np.array(frames)
            frames_idx = [int(x * nframe / depth) for x in range(depth)]
            X.append(np.array(frames_arr[frames_idx]))
            X = np.array(X)
            X = X.transpose((0, 2, 3, 1))
            X = X.reshape((X.shape[0], 32, 32, 20, 1))

            y_pred = model.predict(X)
            y_pred = np.argmax(y_pred,axis=1)[0]
            print(y_pred)

            #server 
            if (y_pred!=0): 
                data={"Gesture_type": y_pred}
                r = requests.post(url="http://localhost:9000/api/gesture/gesture", data=data)


            #####
            pred_label = final_labels[y_pred]
            print(pred_label)
            X = []

        if len(frames) == nframe + skip_frame:
            frames = []

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray,pred_label,(40,40), font, 1,(255,255,255),2)
        cv2.imshow('frame',gray)
        #print (gray)
        os.remove(message)
		
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break;
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()