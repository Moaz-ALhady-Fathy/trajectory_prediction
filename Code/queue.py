import numpy as np
import cv2
import queue
from threading import Thread

# global variables
stop_thread = False             # controls thread execution


def start_capture_thread(camera, queue):
	while True:
		_, img = camera.read()
		queue.put(img)
def get_queue(rate,lenth,camera):
    images=[]
    global stop_thread
    
    frames_queue = queue.Queue(maxsize=0)
    
    # start the capture thread: reads frames from the camera (non-stop) and stores the result in img
    t = Thread(target=start_capture_thread, args=(camera, frames_queue), daemon=True) # a deamon thread is killed when the application exits
    t.start()
    frames = 0
    i=0
    while(True):
        if (frames_queue.empty()):
            continue
        if i % rate !=0 :
            _ = frames_queue.get()
            i+=1
            continue
        # blocks until the entire frame is read
        frames += 1
        # retrieve an image from the queue
        img = frames_queue.get()
        images.append(img)
        if frames==lenth:
            cv2.destroyAllWindows()
            return images
        if frames==lenth:
            break
