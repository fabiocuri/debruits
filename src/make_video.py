import cv2
import numpy as np
import glob

if __name__ == "__main__":

    img_array = []
    
    for filename in glob.glob("../data/output/inference/test_1/*.png"):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        img = img[185:300, 90:565, :]
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter("../data/output/inference/test_1.avi", cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
    for i in range(len(img_array)):
    
        out.write(img_array[i])
        
    out.release()     
    
    # https://www.onlineconverter.com/add-audio-to-video : add audio to this video and add volume 100%
